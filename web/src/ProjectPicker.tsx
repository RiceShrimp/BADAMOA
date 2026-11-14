import { useEffect, useState } from "react";
import { api } from "./api";
import type { Project } from "./catalogTypes";
import { usePolling } from "./ui";

const PROJECT_KEY = "badamoa.project"; // shared with the dataset screen: one "current project" everywhere

function remembered(): string {
  try {
    return localStorage.getItem(PROJECT_KEY) ?? "";
  } catch {
    return "";
  }
}

/** The project the user is working in, remembered across screens and reloads. */
export function useProject() {
  const { data: projects } = usePolling(() => api<Project[]>("/projects"), 30000);
  const [projectId, setProjectId] = useState(remembered);

  useEffect(() => {
    if (projects && projects.length > 0 && !projects.some((p) => p.id === projectId)) setProjectId(projects[0].id);
  }, [projects, projectId]);

  const choose = (id: string) => {
    setProjectId(id);
    try {
      localStorage.setItem(PROJECT_KEY, id);
    } catch {
      /* not remembered across reloads */
    }
  };
  const ready = !!projects?.some((p) => p.id === projectId);
  return { projects, projectId: ready ? projectId : "", choose };
}

export function ProjectPicker({
  projects,
  projectId,
  onChange,
}: {
  projects: Project[] | null;
  projectId: string;
  onChange: (id: string) => void;
}) {
  return (
    <label className="project-picker">
      <span className="eyebrow">프로젝트</span>
      <select value={projectId} onChange={(e) => onChange(e.target.value)} aria-label="프로젝트">
        {projects?.map((p) => (
          <option key={p.id} value={p.id}>
            {p.folder ? `${p.folder} / ` : ""}
            {p.name}
          </option>
        ))}
      </select>
    </label>
  );
}
