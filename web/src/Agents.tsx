import { type FormEvent, useState } from "react";
import { type ApiTokenInfo, api } from "./api";
import { CopyBlock, Icon, timeAgo, useHubUrl, usePolling } from "./ui";

const CAPABILITIES = [
  { icon: "server", title: "노드 조회", text: "어떤 서버가 온라인인지, GPU와 디스크가 얼마나 남았는지" },
  { icon: "activity", title: "작업 실행·추적", text: "작업을 실행하고, 끝날 때까지 기다리고, 로그를 읽고, 취소" },
  { icon: "alert", title: "진단", text: "실패한 작업의 원인 분석, 전체 노드 상태 점검" },
];

export function Agents() {
  const { data: tokens, refresh } = usePolling(() => api<ApiTokenInfo[]>("/tokens"), 10000);
  const [name, setName] = useState("claude-code");
  const [created, setCreated] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mcpUrl = `${useHubUrl()}/mcp`;
  const shown = created ?? "<토큰>";

  const create = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const res = await api<{ token: string }>("/tokens", { body: { name } });
      setCreated(res.token);
      setError(null);
      void refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const revoke = async (id: string) => {
    if (!window.confirm("이 토큰을 폐기할까요? 이 토큰을 쓰는 에이전트는 더 이상 접속할 수 없습니다.")) return;
    await api(`/tokens/${id}`, { method: "DELETE" });
    void refresh();
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>에이전트</h1>
          <p className="muted">
            BADAMOA Hub는 MCP 서버입니다. Claude Code 같은 에이전트를 연결하면 아래 일을 대화로 시킬 수 있습니다.
          </p>
        </div>
      </div>

      <div className="stats three">
        {CAPABILITIES.map((c) => (
          <div className="stat" key={c.title}>
            <div className="eyebrow">
              <Icon name={c.icon} size={14} />
              {c.title}
            </div>
            <p className="cap-text">{c.text}</p>
          </div>
        ))}
      </div>

      <div className="two-col">
        <section className="card">
          <div className="card-head">
            <h3>
              <span className="step-num">1</span>토큰 만들기
            </h3>
          </div>
          <p className="muted">에이전트는 이 토큰을 발급한 사람의 권한으로 동작합니다.</p>
          <form className="inline-form" onSubmit={create}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="토큰 이름" required aria-label="토큰 이름" />
            <button className="primary">토큰 발급</button>
          </form>
          {error && <div className="error">{error}</div>}
          {created && <CopyBlock label="새 토큰 — 지금만 볼 수 있으니 복사해 두세요" text={created} />}
        </section>

        <section className="card">
          <div className="card-head">
            <h3>
              <span className="step-num">2</span>에이전트에 연결하기
            </h3>
          </div>
          <CopyBlock
            label="Claude Code"
            text={`claude mcp add --transport http badamoa ${mcpUrl} --header "Authorization: Bearer ${shown}"`}
          />
          <details>
            <summary>그 외 MCP 클라이언트 (JSON 설정)</summary>
            <CopyBlock
              text={JSON.stringify(
                { mcpServers: { badamoa: { type: "http", url: mcpUrl, headers: { Authorization: `Bearer ${shown}` } } } },
                null,
                2,
              )}
            />
          </details>
          <p className="muted small">연결한 뒤 "BADAMOA 노드 상태 점검해줘"라고 요청해 보세요.</p>
        </section>
      </div>

      <section className="card table-wrap">
        <div className="card-head">
          <h3>발급한 토큰</h3>
        </div>
        <table>
          <thead>
            <tr>
              <th>이름</th>
              <th>토큰</th>
              <th>마지막 사용</th>
              <th>발급</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {tokens?.length === 0 && (
              <tr>
                <td colSpan={5} className="empty">
                  발급한 토큰이 없습니다.
                </td>
              </tr>
            )}
            {tokens?.map((t) => (
              <tr key={t.id}>
                <td>
                  <strong>{t.name}</strong>
                </td>
                <td className="mono muted">{t.prefix}…</td>
                <td>{t.last_used_at ? timeAgo(t.last_used_at) : <span className="muted">사용 전</span>}</td>
                <td className="muted">{timeAgo(t.created_at)}</td>
                <td className="right">
                  <button className="danger" onClick={() => revoke(t.id)}>
                    폐기
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
