import { useEffect, useState } from "react";
import { api } from "./api";
import { Modal } from "./ui";

interface Release {
  version: string;
  date: string | null;
  summary: string;
  sections: { title: string; items: string[] }[];
}

const SEEN_KEY = "badamoa.seenVersion";

/** True until the user has opened the notes for this version (e.g. right after an upgrade). */
export function hasUnseenRelease(version: string): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) !== version;
  } catch {
    return false;
  }
}

export function ReleaseNotes({ version, onClose }: { version: string; onClose: () => void }) {
  const [releases, setReleases] = useState<Release[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(SEEN_KEY, version);
    } catch {
      /* storage unavailable: the "new" marker simply stays */
    }
    api<{ releases: Release[] }>("/release-notes")
      .then((res) => setReleases(res.releases))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [version]);

  return (
    <Modal title="릴리즈 노트" onClose={onClose}>
      {error && <div className="error">{error}</div>}
      {releases?.length === 0 && <p className="muted">릴리즈 노트가 없습니다.</p>}
      <div className="releases">
        {releases?.map((release) => (
          <article key={release.version} className="release">
            <header>
              <h3>v{release.version}</h3>
              {release.version === version && <span className="tag accent">현재 버전</span>}
              <span className="muted small">{release.date}</span>
            </header>
            {release.summary && <p>{release.summary}</p>}
            {release.sections.map((section) => (
              <section key={section.title}>
                <div className="eyebrow">{section.title}</div>
                <ul>
                  {section.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </section>
            ))}
          </article>
        ))}
      </div>
      <div className="modal-actions">
        <button onClick={onClose}>닫기</button>
      </div>
    </Modal>
  );
}
