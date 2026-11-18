import { api } from "./api";
import type { CheckModel } from "./catalogTypes";
import { timeAgo, usePolling } from "./ui";

/** The flagged images of a version: by group and by check (what was let be is left out). */
export interface Problems {
  groups: { missing: number; wrong: number; boxes: number; any: number };
  checks: Record<string, number>;
  let_be: number;
}

/** 검수 kept with a version when its work in progress was confirmed (or, for a work in progress, a preview). */
export interface ReviewRecord {
  recorded: boolean;
  preview: boolean;
  base?: { id: string; version: string } | null;
  before?: Problems | null;
  after?: Problems;
  fixed?: number; // images whose labels changed
  fixed_flagged?: number; // of the images the base version had flagged
  let_be?: number; // images whose flagged labels someone judged right
  model?: { evaluation: string; model: CheckModel | null; flagged: number; fixed: number; let_be: number } | null;
  people?: string[];
  at?: string;
  now?: Problems | null; // a version with no record: its problems now
}

const GROUPS: { id: keyof Problems["groups"]; title: string }[] = [
  { id: "missing", title: "미라벨" },
  { id: "wrong", title: "오라벨" },
  { id: "boxes", title: "박스 오류" },
];

/** Before → after for each group, what was fixed or let be, the model's part, who. */
export function ReviewSummary({ record }: { record: ReviewRecord }) {
  const after = record.after;
  if (!after) return null;
  return (
    <div className="review-record">
      <div className="review-record-groups">
        {GROUPS.map((g) => {
          const was = record.before?.groups[g.id];
          const now = after.groups[g.id];
          return (
            <span key={g.id} className={`review-step ${was != null && now < was ? "better" : ""} ${now > (was ?? now) ? "worse" : ""}`}>
              <span className="muted small">{g.title}</span>
              <strong>
                {was != null && was !== now && <span className="was">{was.toLocaleString()} → </span>}
                {now.toLocaleString()}
              </strong>
            </span>
          );
        })}
      </div>
      <div className="small">
        고친 이미지 <strong>{(record.fixed ?? 0).toLocaleString()}</strong>
        {record.base && record.fixed_flagged != null && <> (그중 {record.base.version}에서 걸렸던 것 {record.fixed_flagged.toLocaleString()})</>}
        {" · "}문제 없음으로 둔 이미지 <strong>{(record.let_be ?? 0).toLocaleString()}</strong>
      </div>
      {record.model && (
        <div className="small">
          {record.model.model ? `${record.model.model.name} v${record.model.model.version}` : "모델"} 기준으로 걸린 이미지{" "}
          {record.model.flagged.toLocaleString()} → 고침 {record.model.fixed.toLocaleString()} · 문제 없음 {record.model.let_be.toLocaleString()} ·
          남음 {(record.model.flagged - record.model.fixed - record.model.let_be).toLocaleString()}
        </div>
      )}
      {(record.people?.length ?? 0) > 0 && (
        <div className="muted small">
          검수한 사람 {record.people?.join(", ")}
          {record.at && !record.preview ? ` · ${timeAgo(record.at)}` : ""}
        </div>
      )}
    </div>
  );
}

/** On a version's screen: the 검수 record it keeps, when it has one. */
export function ReviewRecordCard({ releaseId }: { releaseId: string }) {
  const { data: record } = usePolling(() => api<ReviewRecord>(`/releases/${releaseId}/review`), 0, [releaseId]);
  if (!record?.recorded) return null;
  return (
    <section className="card review-record-card">
      <div className="eyebrow">
        검수 기록 <span className="keep-case">· {record.base ? `${record.base.version}에서 이 버전으로` : "이 버전을 확정할 때"}</span>
        <a className="link small" href={`#/review/${releaseId}`}>
          검수 화면
        </a>
      </div>
      <ReviewSummary record={record} />
    </section>
  );
}
