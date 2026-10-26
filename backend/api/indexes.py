from uuid import UUID

from fastapi import APIRouter, Request, Response

from backend.core.errors import AppError
from backend.schemas import (
    DeleteVectors,
    IndexCreate,
    IndexInfo,
    Insert,
    LoadIndex,
    SavedIndex,
    Search,
    SearchResponse,
    Training,
)

router = APIRouter(prefix="/indexes", tags=["indexes"])


@router.post("", response_model=IndexInfo, status_code=201)
def create(body: IndexCreate, request: Request):
    return request.app.state.indexes.create(body)


@router.get("", response_model=list[IndexInfo])
def listing(request: Request):
    return request.app.state.indexes.list()


@router.get("/saved", response_model=list[SavedIndex])
def saved(request: Request):
    return request.app.state.persistence.list()


@router.post("/load", response_model=IndexInfo, status_code=201)
def load(body: LoadIndex, request: Request):
    return request.app.state.persistence.load(body.archive_id)


@router.get("/{index_id}", response_model=IndexInfo)
def details(index_id: UUID, request: Request):
    with request.app.state.indexes.access(index_id) as entry:
        return entry.info()


@router.delete("/{index_id}", status_code=204)
def delete(index_id: UUID, request: Request):
    request.app.state.indexes.delete(index_id)
    return Response(status_code=204)


@router.post("/{index_id}/train", response_model=IndexInfo)
def train(index_id: UUID, body: Training, request: Request):
    with request.app.state.indexes.access(index_id) as entry:
        if entry.config.type != "ivf":
            raise AppError(422, "unsupported_operation", "Exact indexes do not require training.")
        if (
            len(body.vectors) * entry.engine.dimension * entry.engine.nlist * entry.config.max_iter
            > 100000000
        ):
            raise AppError(
                422,
                "training_limit",
                "Training exceeds the 100 million coordinate-iteration budget.",
            )
        entry.engine.train(body.vectors)
        return entry.info()


@router.post("/{index_id}/vectors", response_model=IndexInfo)
def insert(index_id: UUID, body: Insert, request: Request):
    with request.app.state.indexes.access(index_id) as entry:
        size = entry.engine.size + len(body.vectors)
        if size > 50000 or size * entry.engine.dimension > 2000000:
            raise AppError(422, "capacity", "Index exceeds vector/coordinate capacity.")
        entry.engine.add(body.vectors, body.ids)
        return entry.info()


@router.delete("/{index_id}/vectors", response_model=IndexInfo)
def delete_vectors(index_id: UUID, body: DeleteVectors, request: Request):
    with request.app.state.indexes.access(index_id) as entry:
        entry.engine.delete(body.ids)
        return entry.info()


@router.post("/{index_id}/search", response_model=SearchResponse)
def search(index_id: UUID, body: Search, request: Request):
    return request.app.state.indexes.search(index_id, body)


@router.post("/{index_id}/save", response_model=SavedIndex, status_code=201)
def save(index_id: UUID, request: Request):
    return request.app.state.persistence.save(index_id)
