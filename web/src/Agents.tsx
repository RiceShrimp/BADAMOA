import { type FormEvent, useState } from "react";
import { type ApiTokenInfo, api } from "./api";
import { CopyBlock, timeAgo, usePolling } from "./ui";

export function Agents() {
  const { data: tokens, refresh } = usePolling(() => api<ApiTokenInfo[]>("/tokens"), 10000);
  const [name, setName] = useState("claude-code");
  const [created, setCreated] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mcpUrl = `${window.location.origin}/mcp`;
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
          <h1>에이전트 연결</h1>
          <p className="muted">
            BADAMOA Hub는 MCP 서버입니다. Claude Code 같은 에이전트를 붙이면 노드 조회와 작업 실행을 대화로 할 수 있습니다.
          </p>
        </div>
      </div>

      <section className="card">
        <h3>1. 토큰 만들기</h3>
        <form className="inline-form" onSubmit={create}>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="토큰 이름" required />
          <button className="primary">토큰 발급</button>
        </form>
        {error && <div className="error">{error}</div>}
        {created && (
          <>
            <CopyBlock label="새 토큰 — 지금만 볼 수 있으니 복사해 두세요" text={created} />
          </>
        )}
      </section>

      <section className="card">
        <h3>2. 에이전트에 연결하기</h3>
        <CopyBlock
          label="Claude Code"
          text={`claude mcp add --transport http badamoa ${mcpUrl} --header "Authorization: Bearer ${shown}"`}
        />
        <CopyBlock
          label="그 외 MCP 클라이언트 (JSON 설정)"
          text={JSON.stringify(
            { mcpServers: { badamoa: { type: "http", url: mcpUrl, headers: { Authorization: `Bearer ${shown}` } } } },
            null,
            2,
          )}
        />
        <p className="muted small">연결한 뒤 에이전트에게 "BADAMOA 노드 상태 점검해줘"라고 요청해 보세요.</p>
      </section>

      <section className="card table-wrap">
        <h3>발급한 토큰</h3>
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
                <td>{t.name}</td>
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
