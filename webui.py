import asyncio
import json
import threading
import webbrowser
from contextlib import asynccontextmanager
from queue import SimpleQueue
from typing import Literal

from fastapi import FastAPI, websockets
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils import MaaWorker


class ConfigModel(BaseModel):
    api_key: str

class DeviceModel(BaseModel):
    name: str
    adb_path: str
    address: str
    # 会自动转换为int，不用处理
    screencap_methods: int
    input_methods: int
    config: dict

class TaskModel(BaseModel):
    tasklist: list[Literal["选读文章", "视听学习", "每日答题", "趣味答题"]]

class AppState:
    def __init__(self):
        self.message_conn = SimpleQueue()
        self.child_process = None
        self.worker = None
        self.history_message = []
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    webbrowser.open_new("http://127.0.0.1:8000")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory="page/assets"))

@app.get("/")
async def serve_homepage():
    return FileResponse("page/index.html")

@app.get("/api/settings")
def get_settings():
    with open("./config/config.json") as f:
        config = json.load(f)
    if config["api_key"] != "" and app_state.worker is None:
        app_state.worker = MaaWorker(app_state.message_conn,api_key=config["api_key"])
    return config

@app.post("/api/settings")
def post_settings(config: ConfigModel):
    with open("./config/config.json", "w") as f:
        f.write(config.model_dump_json(indent=4))
    if app_state.worker is None:
        app_state.worker = MaaWorker(app_state.message_conn, api_key=config.api_key)
    return {"status": "success"}

@app.get("/api/get_device")
def get_device():
    if app_state.worker is None:
        return {"status": "failed","message":"MAA未初始化，请先保存设置"}
    devices = app_state.worker.get_device()
    return {"devices": devices}

@app.post("/api/connect_device")
def connect_device(device: DeviceModel):
    if app_state.worker.connect_device(device):
        return {"status": "success"}
    return {"status": "failed"}

@app.post("/api/start")
def start(tasks: TaskModel):
    if app_state.worker is None:
        return {"status": "failed","message":"MAA未初始化，请先保存设置"}
    if app_state.child_process is not None:
        return {"status": "failed","message":"任务已开始"}
    if not app_state.worker.connected:
        return {"status": "failed","message":"请先连接设备"}
    app_state.child_process = threading.Thread(
        target=app_state.worker.task,
        args=(tasks.tasklist,),
        daemon=True
    )
    app_state.child_process.start()
    return {"status": "success"}

@app.post("/api/stop")
def stop():
    if app_state.child_process is None or app_state.worker is None:
        return {"status": "failed","message":"任务未开始"}
    app_state.worker.stop_flag = True
    app_state.child_process = None
    return {"status": "success"}

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: websockets.WebSocket):
    await websocket.accept()
    if app_state.history_message:
        for i in app_state.history_message:
            await websocket.send_text(i)
    while True:
        if not app_state.message_conn.empty():
            data = app_state.message_conn.get_nowait()
            app_state.history_message.append(data)
            if "所有任务完成" in data:
                app_state.child_process.join()
                # 重置状态
                app_state.child_process = None
                # app_state.history_message = []
                await websocket.send_text(data)
                # await websocket.close()
                # break
                continue
            await websocket.send_text(data)
        await asyncio.sleep(0.01)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
