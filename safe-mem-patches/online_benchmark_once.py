from og_ego_prim.utils.monkey_patch import add_monkey_patch
add_monkey_patch()

import argparse
import os
import sys

import omnigibson as og
from omnigibson.macros import gm
import shutil
import time
import torch

from og_ego_prim.benchmark import build_benchmark
from og_ego_prim.models import PlanningAgent
from og_ego_prim.scene_graph_client import (
    DEFAULT_SCENE_GRAPH_PYTHON,
    DEFAULT_SCENE_GRAPH_ROOT,
    SceneGraphWorkerClient,
    find_saved_observation,
)
from og_ego_prim.risk_predictor import (
    DEFAULT_RISK_PROMPT_PATH,
    RiskPredictor,
    RiskPredictorError,
)
from og_ego_prim.utils.constants import BDDLS

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True

parser = argparse.ArgumentParser()
parser.add_argument('--try_id', type=str, default=None)

parser.add_argument('--task', type=str, default=None)
parser.add_argument('--scene', type=str, default=None)
parser.add_argument('--model', type=str, default=None, help="If not local llm, referece to the model_id, if local_llm, referece to the local model path.")
parser.add_argument('--local_llm_serve', action='store_true')
parser.add_argument('--local_serve_ip', type=str, default="")
parser.add_argument('--local_serve_key', type=str, default="sk-123456")
parser.add_argument('--work_dir', type=str, default='./work_dir')

parser.add_argument('--draw_bbox_2d', action='store_true')
parser.add_argument('--use_initial_setup', action='store_true')
parser.add_argument('--use_self_caption', action='store_true')
parser.add_argument('--online_object_sampling', type=bool, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--occlude_risk_source', action='store_true')
parser.add_argument('--use_scene_graph', '--use-scene_graph', action='store_true')
parser.add_argument('--scene_graph_python', type=str, default=DEFAULT_SCENE_GRAPH_PYTHON)
parser.add_argument('--scene_graph_root', type=str, default=DEFAULT_SCENE_GRAPH_ROOT)
parser.add_argument('--scene_graph_camera_index', type=int, default=None)
parser.add_argument('--use_risk_predictor', '--use-risk-predictor', action='store_true')
parser.add_argument('--risk_check_probability', '--risk-check-probability', type=float, default=0.8)
parser.add_argument('--risk_predictor_model', type=str, default='gpt-4o')
parser.add_argument('--risk_prompt_path', type=str, default=DEFAULT_RISK_PROMPT_PATH)
parser.add_argument('--risk_max_replans', type=int, default=10)
parser.add_argument('--risk_random_seed', type=int, default=None)
parser.add_argument('--risk_gate_mode', type=str, default='off',
                    choices=['off', 'pure', 'hybrid'],
                    help="Direction A: 'off' (stochastic eta skip), 'pure' (symbolic gate only), 'hybrid' (gate with LLM fallback)")
parser.add_argument('--max_step', type=int, default=None, help="Override planner step budget (default: len(example_planning) + 10)")

parser.add_argument('--not_eval_process_safety', action='store_true')
parser.add_argument('--not_eval_termination_safety', action='store_true')
parser.add_argument('--not_eval_awareness', action='store_true')
parser.add_argument('--not_eval_execution', action='store_true')
parser.add_argument('--prompt_setting', type=str, default='default')


def online_benchmark_once(
    try_id,
    task: str,
    scene: str,
    model: str,
    local_llm_serve: str, 
    local_serve_ip: str,
    local_serve_key: str,
    work_dir: str,
    draw_bbox_2d: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    online_object_sampling: bool,
    occlude_risk_source: bool,
    use_scene_graph: bool,
    scene_graph_python: str,
    scene_graph_root: str,
    scene_graph_camera_index: int,
    use_risk_predictor: bool,
    risk_check_probability: float,
    risk_predictor_model: str,
    risk_prompt_path: str,
    risk_max_replans: int,
    risk_random_seed: int,
    risk_gate_mode: str,
    max_step: int,
    debug: bool,
    eval_process_safety: bool,
    eval_termination_safety: bool,
    eval_awareness: bool,
    eval_execution: bool,
    prompt_setting: str
):
    if use_risk_predictor and not use_scene_graph:
        raise ValueError('--use_risk_predictor requires --use_scene_graph')
    if use_risk_predictor and not model:
        raise ValueError('--use_risk_predictor requires a VLM planner via --model')
    if not 0.0 <= risk_check_probability <= 1.0:
        raise ValueError('--risk_check_probability must be between 0 and 1')
    if risk_max_replans < 1:
        raise ValueError('--risk_max_replans must be at least 1')
    if risk_gate_mode not in ('off', 'pure', 'hybrid'):
        raise ValueError('--risk_gate_mode must be off, pure, or hybrid')

    benchmark = build_benchmark(
        task=task, 
        scene=scene, 
        ego_view=True,
        draw_bbox_2d=draw_bbox_2d,
        use_initial_setup=use_initial_setup,
        use_self_caption=use_self_caption,
        online_object_sampling=online_object_sampling, 
        debug=debug,
        eval_process_safety=eval_process_safety,
        eval_termination_safety=eval_termination_safety,
        eval_awareness=eval_awareness,
        eval_execution=eval_execution,
    )
    if debug and gm.HEADLESS is False:
        og.sim.enable_viewer_camera_teleoperation()

    benchmark_tag = f'{benchmark.task_name}___{benchmark.scene_name}'
    model_tag = args.model.replace('/', '__') if args.model is not None else 'example'
    
    if not try_id:
        output_dir = os.path.join(work_dir, 'benchmark', benchmark_tag, model_tag)
    else: 
        output_dir = os.path.join(work_dir, 'benchmark', benchmark_tag, f"{try_id}_{model_tag}")
    os.makedirs(output_dir, exist_ok=True)

    scene_graph_client = None
    if use_scene_graph:
        definition_id = benchmark.env_config['task'].get('activity_definition_id', 0)
        bddl_path = os.path.join(
            BDDLS,
            benchmark.task_name,
            f'problem{definition_id}.bddl',
        )
        scene_graph_client = SceneGraphWorkerClient(
            output_dir=os.path.join(output_dir, 'scene_graph'),
            bddl_path=bddl_path,
            python_executable=scene_graph_python,
            scene_graph_root=scene_graph_root,
        ).start()

    risk_predictor = None
    scene_graph_path = os.path.join(output_dir, 'scene_graph', 'scene_graph.txt')

    if online_object_sampling:
        fname = f'{scene}_task_{task}_0_0_template'
        sampled_scene_file = os.path.join(output_dir, f'{fname}.json')
        benchmark.env.task.save_task(path=sampled_scene_file)

    if model or local_llm_serve:
        agent = PlanningAgent(
            task_name=task, 
            scene_name=scene, 
            agent_name=model,
            work_dir=args.work_dir,
            local_llm_serve=local_llm_serve, 
            local_serve_ip=local_serve_ip,  
            local_serve_key=local_serve_key, 
            debug=debug,
            prompt_setting=prompt_setting,
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
            observation_root=output_dir,
        )
        agent.set_tracker(benchmark.tracker)
        if use_risk_predictor:
            shared_client = None
            if not local_llm_serve:
                shared_client = getattr(agent.client, 'client', None)
            risk_predictor = RiskPredictor(
                model=risk_predictor_model,
                probability=risk_check_probability,
                prompt_path=risk_prompt_path,
                random_seed=risk_random_seed,
                client=shared_client,
                gate_mode=risk_gate_mode,
            )
            if shared_client is not None:
                print('[risk] reusing planner API client')
                sys.stdout.flush()
        planner = agent.step(use_obs=True, max_step=(max_step if max_step is not None else len(benchmark._example_planning) + 10))
    else:
        planner = benchmark.get_example_planning()

    initial_step_dir = os.path.join(output_dir, '0_init')
    initial_observations = benchmark.get_surrounding_viewer_obs(save_img=initial_step_dir)
    if scene_graph_client is not None:
        if initial_observations is None:
            benchmark.get_viewer_obs(save_img=os.path.join(initial_step_dir, 'obs.png'))
        camera_index = scene_graph_camera_index
        if camera_index is None:
            camera_index = benchmark.selected_camera_index
        initial_image = find_saved_observation(initial_step_dir, camera_index)
        scene_graph_client.initialize(initial_image, step_name='0_init')
    if occlude_risk_source:
        benchmark.load_fixed_risk_occluder()
    if use_self_caption:
        caption = agent.generate_caption(use_obs=True)
        benchmark.tracker.track_caption(
            content=caption
        )
    if eval_awareness and (model or local_llm_serve):
        awareness = agent.generate_awareness(use_obs=True)
        benchmark.evaluate_awareness(awareness)
    elif prompt_setting == 'v2':
        awareness = agent.generate_awareness(use_obs=True)
        benchmark.tracker.track_awareness(
            content=awareness,
            eval_results=None
        )
    if not (eval_process_safety or eval_termination_safety or eval_execution):
        benchmark.tracker.save_tracking(os.path.join(output_dir, 'report_awareness.json'))
        if scene_graph_client is not None:
            scene_graph_client.close()
        time.sleep(3)
        og.clear()
        return

    executed_step = 0
    consecutive_risk_replans = 0
    for plan in planner:
        if risk_predictor is not None:
            try:
                decision = risk_predictor.predict(
                    scene_graph_path=scene_graph_path,
                    action=plan['action'],
                )
            except (RiskPredictorError, ValueError) as error:
                benchmark.tracker.track_risk_prediction(
                    action=plan['action'],
                    checked=True,
                    safe=False,
                    outcome='error',
                    risk_source=str(error),
                    replan_attempt=consecutive_risk_replans,
                )
                benchmark.tracker.track_error(
                    action=plan['action'],
                    err_type=error.__class__.__name__,
                    msg=str(error),
                )
                benchmark.tracker.track_termination(
                    reason='risk_predictor_error',
                    type=error.__class__.__name__,
                    msg=str(error),
                )
                print(f'[risk] predictor error; action blocked: {error}')
                sys.stdout.flush()
                break

            risk_record = decision.to_dict()
            risk_record.update(
                action=plan['action'],
                replan_attempt=consecutive_risk_replans,
                scene_graph=scene_graph_path,
            )
            if decision.safe:
                risk_record['outcome'] = (
                    'allowed' if decision.checked else 'probability_bypass'
                )
                benchmark.tracker.track_risk_prediction(**risk_record)
                agent.accept_plan()
                consecutive_risk_replans = 0
                print(
                    f"[risk] action allowed ({risk_record['outcome']}): "
                    f"{plan['action']} - {decision.risk_source}"
                )
                sys.stdout.flush()
            else:
                consecutive_risk_replans += 1
                risk_record['outcome'] = 'rejected'
                risk_record['replan_attempt'] = consecutive_risk_replans
                benchmark.tracker.track_risk_prediction(**risk_record)
                agent.reject_plan(
                    plan,
                    risk_source=decision.risk_source,
                )
                print(
                    f'[risk] action rejected; requesting replan '
                    f'({consecutive_risk_replans}/{risk_max_replans})'
                )
                sys.stdout.flush()
                if consecutive_risk_replans >= risk_max_replans:
                    benchmark.tracker.track_termination(
                        reason='risk_replan_exhausted',
                        type='RiskPredictorError',
                        msg=(
                            f'no safe action accepted after '
                            f'{risk_max_replans} replans'
                        ),
                    )
                    break
                continue

        if benchmark.execute_plan(plan) is False:
            break
        executed_step += 1
        step_tag = f'{executed_step}_' + plan['action'].replace('(', '__').replace(')', '__')
        step_dir = os.path.join(output_dir, step_tag)
        step_observations = benchmark.get_surrounding_viewer_obs(save_img=step_dir)
        if scene_graph_client is not None:
            if step_observations is None:
                benchmark.get_viewer_obs(save_img=os.path.join(step_dir, 'obs.png'))
            camera_index = scene_graph_camera_index
            if camera_index is None:
                camera_index = benchmark.selected_camera_index
            observation_path = find_saved_observation(step_dir, camera_index)
            scene_graph_client.update(
                observation_path,
                action=plan['action'],
                step_name=step_tag,
            )

    benchmark.termination_evaluation()
    benchmark.tracker.save_tracking(os.path.join(output_dir, 'report.json'))
    if scene_graph_client is not None:
        scene_graph_client.close()
    
    if online_object_sampling:
        if benchmark.tracker.termination['reason'] == 'done' and benchmark.tracker.goal_condition['execution_goal_condition']['eval']: 
            normal_scene_file = os.path.join(work_dir, "..", "data", "scenes", scene, "json", f'{fname}.json')
            shutil.copyfile(sampled_scene_file, normal_scene_file)
        else:
            os.remove(sampled_scene_file)

    time.sleep(3)
    og.clear()


if __name__ == "__main__":
    args = parser.parse_args()
    print(f'args: {args}')
    sys.stdout.flush()

    online_benchmark_once(
        try_id=args.try_id,
        task=args.task,
        scene=args.scene,
        model=args.model,
        local_llm_serve=args.local_llm_serve,
        local_serve_ip=args.local_serve_ip,
        local_serve_key=args.local_serve_key,
        prompt_setting=args.prompt_setting,
        work_dir=args.work_dir,
        draw_bbox_2d=args.draw_bbox_2d,
        use_initial_setup=args.use_initial_setup,
        use_self_caption=args.use_self_caption,
        online_object_sampling=args.online_object_sampling,
        occlude_risk_source=args.occlude_risk_source,
        use_scene_graph=args.use_scene_graph,
        scene_graph_python=args.scene_graph_python,
        scene_graph_root=args.scene_graph_root,
        scene_graph_camera_index=args.scene_graph_camera_index,
        use_risk_predictor=args.use_risk_predictor,
        risk_check_probability=args.risk_check_probability,
        risk_predictor_model=args.risk_predictor_model,
        risk_prompt_path=args.risk_prompt_path,
        risk_max_replans=args.risk_max_replans,
        risk_random_seed=args.risk_random_seed,
        risk_gate_mode=args.risk_gate_mode,
        max_step=args.max_step,
        debug=args.debug,
        eval_process_safety=(not args.not_eval_process_safety),
        eval_termination_safety=(not args.not_eval_termination_safety),
        eval_awareness=(not args.not_eval_awareness),
        eval_execution=(not args.not_eval_execution),
    )
