import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { writeFileSync } from "fs";
import { resolve } from "path";

// Plugin that recreates .gitkeep after emptyOutDir clears the assets dir.
const preserveGitkeep = () => ({
  name: "preserve-gitkeep",
  closeBundle() {
    const gitkeep = resolve(__dirname, "../crates/ai-engine-web/assets/.gitkeep");
    writeFileSync(gitkeep, "");
  },
});

export default defineConfig({
  plugins: [react(), preserveGitkeep()],
  build: {
    outDir: "../crates/ai-engine-web/assets",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/v1": "http://localhost:8080",
      "/cluster": "http://localhost:8080",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
