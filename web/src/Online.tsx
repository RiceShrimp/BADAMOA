import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { timeAgo, usePolling } from "./ui";

/** Someone using the Hub now: a browser session or an API token used in the last three minutes. */
export interface OnlineUser {
  id: string;
  username: string;
  role: string;
  last_seen_at: string;
  web: number; // open browser sessions
  api: number; // API tokens in use (MCP, scripts)
  ips: string[];
}

/** Top bar, admins: who is using the Hub now, and from where. */
export function OnlineTray({ me }: { me: string }) {
  const { data: people } = usePolling(() => api<OnlineUser[]>("/users/online"), 30000);
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  if (!people) return null;
  return (
    <div className="import-tray" ref={box}>
      <button className="pill ok" onClick={() => setOpen(!open)} title="지금 BADAMOA를 쓰는 사람 (최근 3분)">
        <i />
        접속 {people.length}명
      </button>
      {open && (
        <div className="import-panel">
          <div className="import-panel-head">
            <strong>지금 접속 중</strong>
            <span className="muted small">최근 3분 안에 화면을 열어 두었거나 API · MCP로 쓴 사람</span>
          </div>
          {people.map((p) => (
            <div key={p.id} className="import-row">
              <div className="import-row-head">
                <strong>
                  {p.username}
                  {p.id === me && <span className="tag">나</span>}
                </strong>
                <span className="muted small">
                  {p.role === "admin" ? "관리자" : "멤버"} · {timeAgo(p.last_seen_at)}
                </span>
              </div>
              <div className="muted small">
                {[p.web ? `웹 ${p.web}곳` : "", p.api ? `API 토큰 ${p.api}개` : ""].filter(Boolean).join(" · ")}
                {p.ips.length > 0 && ` · ${p.ips.join(", ")}`}
              </div>
            </div>
          ))}
          <a className="small" href="#/settings/users" onClick={() => setOpen(false)}>
            사용자 관리
          </a>
        </div>
      )}
    </div>
  );
}
