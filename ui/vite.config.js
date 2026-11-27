import { fileURLToPath } from 'node:url';
import { defineConfig, loadEnv } from 'vite';

const projectRoot = fileURLToPath(new URL('..', import.meta.url));

export function createDashboardConfig(env) {
  return {
    envDir: projectRoot,
    plugins: [{
      name: 'dashboard-environment',
      configureServer(server) {
        server.middlewares.use('/api/dashboard/config', (_req, res) => {
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ realPrometheusConfigured: Boolean(env.REAL_PROM_URL), agentConfigured: Boolean(env.AGENT_URL) }));
        });
      },
    }],
    server: {
      proxy: {
        '/api/sim': { target: env.SIM_URL || 'http://localhost:8080', rewrite: path => path.replace(/^\/api\/sim/, '') },
        ...(env.REAL_PROM_URL ? { '/api/real-prom': { target: env.REAL_PROM_URL, rewrite: path => path.replace(/^\/api\/real-prom/, '') } } : {}),
        '/api/prom': { target: env.PROM_URL || 'http://localhost:9090', rewrite: path => path.replace(/^\/api\/prom/, '') },
        ...(env.AGENT_URL ? { '/api/agent': { target: env.AGENT_URL, rewrite: path => path.replace(/^\/api\/agent/, '') } } : {}),
      },
    },
  };
}

export default defineConfig(({ mode }) => createDashboardConfig(loadEnv(mode, projectRoot, '')));
