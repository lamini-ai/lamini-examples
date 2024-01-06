from playground.util.get_config import create_config
from playground.fastapi.chat_backend import app as api_app

import uvicorn
from uvicorn.supervisors import ChangeReload

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import asyncio
import sys
import os

config = create_config()

reload_dirs = [
    os.path.join(config["playground"]["home_path"], "scripts"),
    os.path.join(config["playground"]["home_path"], "playground"),
    os.path.join(config["playground"]["home_path"], "src"),
]


async def create_api(port):
    server_config = uvicorn.Config(
        "main:api_app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(server_config)
    await server.serve()


async def main():
    task1 = asyncio.create_task(create_api(5001))
    done, pending = await asyncio.wait(
        [
            task1,
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )

    print("done", done)
    print("pending", pending)
    for pending_task in pending:
        pending_task.cancel("Another service died, server is shutting down")

def build_frontend():
    os.chdir(config["playground"]["home_path"])
    os.system("yarn build-development")

    frontend_path = config["playground"]["home_path"] + "/build"

    if os.path.exists(frontend_path):
        templates = Jinja2Templates(directory=frontend_path)

        api_app.mount(
            "/static",
            StaticFiles(directory=frontend_path + "/static", html=True),
            name="static",
        )

        @api_app.route("/")
        @api_app.route("/playground")
        async def catch_all(request: Request, full_path: str = "/"):
            return templates.TemplateResponse("index.html", {"request": request})


def run_server(sockets):
    build_frontend()
    asyncio.run(main())


def setup_and_run_server():
    server_config = uvicorn.Config(
        "main:run_server",
        host="0.0.0.0",
        port=5001,
        log_level="info",
        reload_dirs=reload_dirs,
        reload_excludes=[],
        reload=True,
        reload_includes=["**/*.py", "**/*.yaml", "**/*.tsx", "**/*.ts"],
    )

    sock = server_config.bind_socket()

    supervisor = ChangeReload(
        server_config, target=run_server, sockets=[5001]
    )

    supervisor.run()


if __name__ == "__main__":
    try:
        setup_and_run_server()
    except Exception as e:
        print(e)
        sys.exit(0)
