import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import sys


DEFAULT_SCENE_GRAPH_ROOT = str(Path(__file__).resolve().parents[1] / "scene_graph")
DEFAULT_SCENE_GRAPH_PYTHON = sys.executable


def find_saved_observation(step_dir, camera_index=None):
    step_dir = Path(step_dir)
    if camera_index is not None:
        selected = step_dir / f"obs_{camera_index}.png"
        if selected.is_file():
            return selected
        raise FileNotFoundError(
            f"scene_graph camera observation not found: {selected}"
        )

    single_view = step_dir / "obs.png"
    if single_view.is_file():
        return single_view

    observations = sorted(step_dir.glob("obs_*.png"))
    if len(observations) == 1:
        return observations[0]
    if not observations:
        raise FileNotFoundError(f"No saved PNG observation found in {step_dir}")
    raise RuntimeError(
        f"Multiple observations found in {step_dir}; pass --scene_graph_camera_index"
    )


class SceneGraphWorkerClient:
    def __init__(
        self,
        output_dir,
        bddl_path,
        python_executable=DEFAULT_SCENE_GRAPH_PYTHON,
        scene_graph_root=DEFAULT_SCENE_GRAPH_ROOT,
        timeout=3600,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.bddl_path = Path(bddl_path).resolve()
        self.python_executable = Path(python_executable).resolve()
        self.scene_graph_root = Path(scene_graph_root).resolve()
        self.timeout = timeout
        self.process = None

    def start(self):
        if self.process is not None:
            raise RuntimeError("scene_graph worker has already been started")
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"scene_graph Python executable not found: {self.python_executable}"
            )
        if not self.scene_graph_root.is_dir():
            raise FileNotFoundError(f"scene_graph repository not found: {self.scene_graph_root}")
        if not self.bddl_path.is_file():
            raise FileNotFoundError(f"Task BDDL file not found: {self.bddl_path}")
        if self.output_dir.exists():
            raise FileExistsError(
                f"scene_graph output already exists; use a new --try_id: {self.output_dir}"
            )

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        # Direction A: pin the scene-graph worker (SAM2) to a dedicated GPU.
        # Isaac Sim's kit may override CUDA_VISIBLE_DEVICES in the parent process,
        # so set it explicitly here for the SAM2 subprocess instead of inheriting.
        environment["CUDA_VISIBLE_DEVICES"] = environment.get(
            "SCENE_GRAPH_GPU", "0"
        )
        package_parent = str(self.scene_graph_root.parent)
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_parent
            if not current_pythonpath
            else package_parent + os.pathsep + current_pythonpath
        )

        command = [
            str(self.python_executable),
            "-u",
            "-m",
            "scene_graph.online_worker",
            "--output-dir",
            str(self.output_dir),
            "--bddl",
            str(self.bddl_path),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.scene_graph_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        atexit.register(self.close)
        ready = self._read_response()
        if not ready.get("ok") or ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"scene_graph worker failed to start: {ready}")
        print(f'[scene_graph] worker ready: {ready["output_dir"]}')
        return self

    def _read_response(self):
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("scene_graph worker is not running")
        readable, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not readable:
            raise TimeoutError(
                f"scene_graph worker did not respond within {self.timeout} seconds"
            )
        line = self.process.stdout.readline()
        if not line:
            return_code = self.process.poll()
            raise RuntimeError(
                f"scene_graph worker exited without a response (return code {return_code})"
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                f'scene_graph {response.get("error_type", "error")}: '
                f'{response.get("error", "unknown failure")}'
            )
        return response

    def _request(self, payload):
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("scene_graph worker is not running")
        if self.process.poll() is not None:
            raise RuntimeError(
                f"scene_graph worker has exited with code {self.process.returncode}"
            )
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        print(
            f'[scene_graph] frame {response.get("frame_index")} complete: '
            f'{response.get("scene_graph")}'
        )
        return response

    def initialize(self, image_path, step_name="0_init"):
        return self._request(
            {
                "command": "initialize",
                "image_path": str(Path(image_path).resolve()),
                "step_name": step_name,
            }
        )

    def update(self, image_path, action, step_name):
        return self._request(
            {
                "command": "update",
                "image_path": str(Path(image_path).resolve()),
                "action": action,
                "step_name": step_name,
            }
        )

    def close(self):
        process = self.process
        if process is None:
            return
        self.process = None
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"command": "close"}) + "\n")
                process.stdin.flush()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            try:
                atexit.unregister(self.close)
            except Exception:
                pass
