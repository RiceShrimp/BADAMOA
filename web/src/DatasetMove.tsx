import { useEffect, useMemo, useState, type FormEvent } from "react";
import { api } from "./api";
import type { Project } from "./catalogTypes";
import { Modal } from "./ui";

/** What moving a dataset takes along and leaves (GET /api/datasets/{id}/move). */
interface MovePlan {
  dataset: { id: string; name: string; task: string };
  project: { id: string; name: string };
  versions: number;
  trashed: number;
  draft: boolean;
  items: number;
  shared: number; // also in another dataset of this project: copied
  label_sets: number;
  pipelines: number;
  stays: { runs: number; models: number; evaluations: number; sweeps: number };
  busy: string[];
  target?: { id: string; name: string; name_taken: boolean; known: number };
}

export interface MovedDataset {
  id: string;
  name: string;
  project: { id: string; name: string };
  release_id: string | null;
}

const joined = (parts: (string | false | 0 | null | undefined)[]) => parts.filter(Boolean).join(" · ");

/** Move a dataset (every version, its work in progress and labeling sets) to another project. */
export function MoveDatasetDialog({
  dataset,
  projectId,
  onClose,
  onMoved,
}: {
  dataset: { id: string; name: string };
  projectId: string;
  onClose: () => void;
  onMoved: (moved: MovedDataset) => void;
}) {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [target, setTarget] = useState("");
  const [plan, setPlan] = useState<MovePlan | null>(null);
  const [name, setName] = useState(dataset.name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api<Project[]>("/projects").then(setProjects, (e: Error) => setError(e.message));
  }, []);
  useEffect(() => {
    setPlan(null);
    const query = target ? `?project_id=${encodeURIComponent(target)}` : "";
    void api<MovePlan>(`/datasets/${dataset.id}/move${query}`).then(setPlan, (e: Error) => setError(e.message));
  }, [dataset.id, target]);

  const others = useMemo(
    () =>
      (projects ?? [])
        .filter((p) => p.id !== projectId)
        .sort((a, b) => (a.folder || "￿").localeCompare(b.folder || "￿") || a.name.localeCompare(b.name)),
    [projects, projectId],
  );
  const there = plan?.target?.id === target ? plan.target : undefined;
  // the other project has a dataset of this name: it goes under another one
  const sameName = !!there?.name_taken && name.trim() === dataset.name;
  const ready = !!there && !!plan && plan.busy.length === 0 && !!name.trim() && !sameName && !busy;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!ready) return;
    setBusy(true);
    setError(null);
    try {
      const body = { project_id: target, ...(name.trim() !== dataset.name ? { name: name.trim() } : {}) };
      onMoved(await api<MovedDataset>(`/datasets/${dataset.id}/move`, { method: "POST", body }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <Modal title="다른 프로젝트로 옮기기" onClose={onClose}>
      <form onSubmit={submit} className="move-dataset">
        <p className="muted small">
          <strong>{dataset.name}</strong>의 모든 버전, 작업 중, 라벨, 라벨링 묶음과 파이프라인을 다른 프로젝트로 옮깁니다. 이미지 파일은
          그대로 있습니다.
        </p>
        <label>
          옮길 프로젝트
          <select value={target} onChange={(e) => setTarget(e.target.value)} required autoFocus>
            <option value="">프로젝트 고르기</option>
            {others.map((p) => (
              <option key={p.id} value={p.id}>
                {p.folder ? `${p.folder} / ` : ""}
                {p.name}
              </option>
            ))}
          </select>
          {projects && others.length === 0 && (
            <span className="muted small field-hint">다른 프로젝트가 없습니다. 데이터셋 화면에서 프로젝트를 먼저 만드세요.</span>
          )}
        </label>
        {target && (
          <label>
            그 프로젝트에서의 이름
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={128} required />
            {sameName && (
              <span className="field-hint error-text small">
                {there!.name}에 {dataset.name} 데이터셋이 이미 있습니다(휴지통에 있는 것 포함). 다른 이름을 쓰세요.
              </span>
            )}
          </label>
        )}

        {plan ? (
          <div className="move-plan">
            <div>
              <span className="eyebrow">함께 옮겨짐</span>
              <p>
                {joined([
                  `버전 ${plan.versions}개`,
                  plan.draft && "작업 중",
                  plan.trashed > 0 && `휴지통의 버전 ${plan.trashed}개`,
                  `이미지 ${plan.items.toLocaleString()}장`,
                  plan.label_sets > 0 && `라벨링 묶음 ${plan.label_sets}개`,
                  plan.pipelines > 0 && `파이프라인 ${plan.pipelines}개`,
                ])}
              </p>
              {plan.shared > 0 && (
                <p className="muted small">
                  이 프로젝트의 다른 데이터셋도 쓰는 이미지 {plan.shared.toLocaleString()}장은 복사됩니다. 다른 데이터셋은 그대로입니다.
                </p>
              )}
              {there && there.known > 0 && (
                <p className="muted small">
                  {there.name}에 이미 있는 이미지 {there.known.toLocaleString()}장은 그쪽 이미지에 이 데이터셋의 라벨을 붙여 씁니다.
                </p>
              )}
              {plan.pipelines > 0 && (
                <p className="muted small">파이프라인은 옮긴 뒤 그 프로젝트에 모델을 만들고, 이름으로 고른 모델도 그 프로젝트에서 찾습니다.</p>
              )}
            </div>
            {(plan.stays.runs > 0 || plan.stays.models > 0 || plan.stays.evaluations > 0 || plan.stays.sweeps > 0) && (
              <div>
                <span className="eyebrow">{plan.project.name}에 남음</span>
                <p>
                  {joined([
                    plan.stays.runs > 0 && `학습 ${plan.stays.runs}개`,
                    plan.stays.sweeps > 0 && `탐색 ${plan.stays.sweeps}개`,
                    plan.stays.models > 0 && `모델 ${plan.stays.models}개`,
                    plan.stays.evaluations > 0 && `평가 ${plan.stays.evaluations}개`,
                  ])}
                </p>
                <p className="muted small">
                  만든 프로젝트에 그대로 남고 옮긴 버전을 계속 가리킵니다. 옮긴 뒤 새로 학습하면 그 프로젝트의 학습 · 모델이 됩니다.
                </p>
              </div>
            )}
          </div>
        ) : (
          !error && <p className="muted small">살펴보는 중…</p>
        )}
        {plan && plan.busy.length > 0 && (
          <div className="error">
            이 데이터셋에서 {plan.busy.join(", ")}가 진행 중이라 옮길 수 없습니다. 끝나거나 멈춘 뒤에 옮기세요.
          </div>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={!ready}>
            {busy ? "옮기는 중…" : "옮기기"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
