import { Icon } from "./ui";

export interface StoredFile {
  name: string; // may carry the folders it was written in: "weights/last.pt"
  kind: "image" | "file";
  bytes: number;
  url: string;
}

export function fileSize(bytes: number): string {
  if (bytes <= 0) return "0 KB";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024)).toLocaleString()} KB`;
}

const baseName = (name: string) => name.slice(name.lastIndexOf("/") + 1);

/** Files as they were written: the ones at the top first, then one fold per folder. */
export function FileList({ files }: { files: StoredFile[] }) {
  const folders = new Map<string, StoredFile[]>();
  for (const file of files) {
    const folder = file.name.slice(0, Math.max(0, file.name.lastIndexOf("/")));
    folders.set(folder, [...(folders.get(folder) ?? []), file]);
  }
  return (
    <>
      {[...folders.keys()].sort().map((folder) =>
        folder === "" ? (
          <FileGroup key="" files={folders.get(folder) ?? []} />
        ) : (
          <details className="file-folder" key={folder}>
            <summary>
              {folder}/ <span className="muted">({folders.get(folder)?.length})</span>
            </summary>
            <FileGroup files={folders.get(folder) ?? []} />
          </details>
        ),
      )}
    </>
  );
}

function FileGroup({ files }: { files: StoredFile[] }) {
  const pictures = files.filter((f) => f.kind === "image");
  const documents = files.filter((f) => f.kind !== "image");
  return (
    <>
      {pictures.length > 0 && (
        <div className="run-pictures">
          {pictures.map((f) => (
            <a key={f.name} href={f.url} target="_blank" rel="noreferrer" title="새 창에서 크게 보기">
              <img src={f.url} alt={f.name} loading="lazy" />
              <span>{baseName(f.name)}</span>
            </a>
          ))}
        </div>
      )}
      {documents.map((f) => (
        <a key={f.name} className="run-file" href={f.url} download>
          <Icon name="copy" size={14} />
          {baseName(f.name)} <span className="muted small">{fileSize(f.bytes)}</span>
        </a>
      ))}
    </>
  );
}
