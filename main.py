from fastapi import FastAPI


app = FastAPI(title="WhyLog FastAPI", version="1.0.0")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "FastAPI server is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}