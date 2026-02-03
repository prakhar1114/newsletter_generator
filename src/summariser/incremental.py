from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from summariser.config import CENTROID_ASSIGN_MIN_COSINE, COLLECTION_NAME
from summariser.ingest import IngestResult, ingest_url
from summariser.reporting import generate_compiled_report
from summariser.vectordb_client import (
    centroid_points,
    points_in_cluster,
    retrieve_points,
    set_payload_for_points,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IncrementalUpdateResult:
    ingested: list[IngestResult]
    assigned: dict[str, int]  # point_id -> cluster_id
    noise_point_ids: list[str]
    touched_clusters: list[int]
    new_centroid_by_cluster: dict[int, str]  # cluster_id -> point_id
    report_path: str | None


def _as_vector(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


async def incremental_update_and_generate_report(
    *,
    urls: list[str],
    collection_name: str = COLLECTION_NAME,
    min_cosine: float = CENTROID_ASSIGN_MIN_COSINE,
    report_model: str = "gpt-5-mini",
    report_system_prompt: str | None = None,
    centroid_limit: int = 5000,
    cluster_limit: int = 5000,
) -> IncrementalUpdateResult:
    """
    End-to-end incremental update:
      1) ingest URLs (adds new points with default payload cluster_id=-1,is_centroid=false)
      2) load existing centroids (is_centroid=true)
      3) assign new points to nearest centroid cluster or noise (-1)
      4) for any touched cluster, recompute the centroid article and update is_centroid flags
      5) generate a new compiled report from stored centroids
    """
    t0 = dt.datetime.now(dt.timezone.utc)
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        raise ValueError("urls must be non-empty")

    logger.info("[incr] urls=%s min_cosine=%.3f collection=%r", len(urls), float(min_cosine), collection_name)

    # 1) ingest all urls
    ingested: list[IngestResult] = []
    failures: list[tuple[str, str]] = []
    for u in urls:
        try:
            res = await ingest_url(u)
        except Exception as e:
            failures.append((u, repr(e)))
            logger.exception("[incr] ingest failed url=%r", u)
        else:
            ingested.append(res)
            logger.info("[incr] ingested url=%r file_id=%s point_id=%s", u, res.file_id, res.point_id)
    if failures:
        logger.info("[incr] ingest failures=%s", len(failures))
    if not ingested:
        raise RuntimeError("No URLs ingested successfully; aborting incremental update")

    new_point_ids = [r.point_id for r in ingested]

    # 2) load centroid vectors + their cluster ids
    c_points = centroid_points(
        collection_name=collection_name,
        limit=centroid_limit,
        with_vectors=True,
        with_payload=True,
    )
    centroid_ids: list[str] = []
    centroid_cluster_ids: list[int] = []
    centroid_vectors: list[np.ndarray] = []
    for p in c_points:
        payload = p.payload or {}
        cid = int(payload.get("cluster_id", -1))
        if cid == -1:
            continue
        v = getattr(p, "vector", None)
        if v is None:
            continue
        centroid_ids.append(str(p.id))
        centroid_cluster_ids.append(cid)
        centroid_vectors.append(_as_vector(v))

    if centroid_vectors:
        centroid_mat = np.stack(centroid_vectors, axis=0)  # (C, D)
        logger.info("[incr] loaded centroids=%s", centroid_mat.shape[0])
    else:
        centroid_mat = np.empty((0, 0), dtype=np.float32)
        logger.info("[incr] loaded centroids=0 (all new points will become noise)")

    # 3) assign each new point to nearest centroid, or noise
    retrieved = retrieve_points(new_point_ids, collection_name=collection_name, with_vectors=True, with_payload=True)
    id_to_vec: dict[str, np.ndarray] = {}
    for p in retrieved:
        v = getattr(p, "vector", None)
        if v is None:
            continue
        id_to_vec[str(p.id)] = _as_vector(v)

    assigned: dict[str, int] = {}
    noise_point_ids: list[str] = []
    touched_clusters_set: set[int] = set()

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    for pid in new_point_ids:
        vec = id_to_vec.get(pid)
        if vec is None or centroid_mat.size == 0:
            assigned_cluster = -1
            best_sim = float("nan")
            best_centroid = None
        else:
            # vectors are normalized at ingest; dot product approximates cosine similarity.
            sims = centroid_mat @ vec
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_centroid = centroid_ids[best_idx]
            assigned_cluster = int(centroid_cluster_ids[best_idx]) if best_sim >= float(min_cosine) else -1

        assigned[pid] = assigned_cluster
        if assigned_cluster == -1:
            noise_point_ids.append(pid)
            logger.info("[incr] assign point_id=%s -> noise (-1) best_sim=%s", pid, f"{best_sim:.3f}" if best_sim == best_sim else "nan")
            set_payload_for_points(
                [pid],
                {"cluster_id": -1, "is_centroid": False, "assigned_at": now_iso},
                collection_name=collection_name,
            )
        else:
            touched_clusters_set.add(assigned_cluster)
            logger.info(
                "[incr] assign point_id=%s -> cluster_id=%s best_sim=%.3f centroid_point_id=%s",
                pid,
                assigned_cluster,
                best_sim,
                best_centroid,
            )
            set_payload_for_points(
                [pid],
                {
                    "cluster_id": int(assigned_cluster),
                    "is_centroid": False,
                    "assigned_at": now_iso,
                    "assigned_centroid_point_id": str(best_centroid),
                },
                collection_name=collection_name,
            )

    touched_clusters = sorted(touched_clusters_set)
    logger.info("[incr] touched_clusters=%s noise=%s", len(touched_clusters), len(noise_point_ids))

    # 4) recompute centroid article for each touched cluster and update is_centroid flags
    new_centroid_by_cluster: dict[int, str] = {}
    for cluster_id in touched_clusters:
        pts = points_in_cluster(
            cluster_id,
            collection_name=collection_name,
            limit=cluster_limit,
            with_vectors=True,
            with_payload=True,
        )
        if not pts:
            logger.info("[incr] cluster_id=%s has no points; skip centroid recompute", cluster_id)
            continue

        cluster_point_ids: list[str] = []
        cluster_vectors: list[np.ndarray] = []
        old_centroid_ids: list[str] = []
        for p in pts:
            pid = str(p.id)
            payload = p.payload or {}
            if bool(payload.get("is_centroid")):
                old_centroid_ids.append(pid)
            v = getattr(p, "vector", None)
            if v is None:
                continue
            cluster_point_ids.append(pid)
            cluster_vectors.append(_as_vector(v))

        if not cluster_vectors:
            logger.info("[incr] cluster_id=%s has no vectors; skip centroid recompute", cluster_id)
            continue

        mat = np.stack(cluster_vectors, axis=0)  # (M, D)
        centroid_vec = mat.mean(axis=0)
        dists = np.linalg.norm(mat - centroid_vec, axis=1)
        best_local = int(np.argmin(dists))
        new_centroid_id = cluster_point_ids[best_local]

        # Only update if centroid actually changes. (Avoid regenerating report on no-op runs.)
        centroid_changed = not (len(old_centroid_ids) == 1 and old_centroid_ids[0] == new_centroid_id)
        if not centroid_changed:
            logger.info("[incr] cluster_id=%s centroid_unchanged centroid_point_id=%s points=%s", cluster_id, new_centroid_id, mat.shape[0])
            continue

        # Update payloads: old centroids -> false, new centroid -> true
        if old_centroid_ids:
            set_payload_for_points(
                old_centroid_ids,
                {"is_centroid": False, "centroid_updated_at": now_iso},
                collection_name=collection_name,
            )
        set_payload_for_points(
            [new_centroid_id],
            {"is_centroid": True, "centroid_updated_at": now_iso},
            collection_name=collection_name,
        )
        new_centroid_by_cluster[int(cluster_id)] = str(new_centroid_id)
        logger.info(
            "[incr] cluster_id=%s centroid_switch old=%s new=%s points=%s",
            cluster_id,
            old_centroid_ids[:3] + (["..."] if len(old_centroid_ids) > 3 else []),
            new_centroid_id,
            mat.shape[0],
        )

    report_path: str | None = None
    if new_centroid_by_cluster:
        # 5) regenerate report from stored centroids
        logger.info(
            "[incr] centroid_changes=%s; generating report from stored centroids ...",
            len(new_centroid_by_cluster),
        )
        _report_md, out_path = generate_compiled_report(
            model=report_model,
            system_prompt=report_system_prompt,
            collection_name=collection_name,
            use_stored_centroids=True,
        )
        report_path = str(out_path)
        logger.info(
            "[incr] report_written=%s elapsed_s=%.2f",
            report_path,
            (dt.datetime.now(dt.timezone.utc) - t0).total_seconds(),
        )
    else:
        logger.info(
            "[incr] no centroid changes detected; skipping report regeneration elapsed_s=%.2f",
            (dt.datetime.now(dt.timezone.utc) - t0).total_seconds(),
        )

    return IncrementalUpdateResult(
        ingested=ingested,
        assigned=assigned,
        noise_point_ids=noise_point_ids,
        touched_clusters=touched_clusters,
        new_centroid_by_cluster=new_centroid_by_cluster,
        report_path=report_path,
    )

