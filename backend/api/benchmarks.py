from uuid import UUID

from fastapi import APIRouter, Request

from backend.schemas import JobInfo, QuickRun, RunInfo, RunList, RunResults

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("/runs", response_model=RunList)
def listing(request: Request):
    return request.app.state.reports.list()


@router.get("/runs/{run_id}", response_model=RunInfo)
def details(run_id: UUID, request: Request):
    return request.app.state.reports.get(run_id).run


@router.get("/runs/{run_id}/results", response_model=RunResults)
def results(run_id: UUID, request: Request):
    return request.app.state.reports.get(run_id)


@router.post("/runs", response_model=JobInfo, status_code=202)
def start(body: QuickRun, request: Request):
    return request.app.state.jobs.submit(body)


@router.get("/jobs", response_model=list[JobInfo])
def jobs(request: Request):
    return request.app.state.jobs.list()


@router.get("/jobs/{job_id}", response_model=JobInfo)
def status(job_id: UUID, request: Request):
    return request.app.state.jobs.get(job_id)
