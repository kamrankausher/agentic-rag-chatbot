import uvicorn

from app.config import HOST, PORT


def main():
    uvicorn.run(
        "app.api:app",
        host=HOST,
        port=PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()