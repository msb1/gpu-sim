import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDashboardConfig } from '../vite.config.js';

function exposedConfig(config) {
  let handler;
  config.plugins[0].configureServer({ middlewares: { use(path, fn) {
    assert.equal(path, '/api/dashboard/config');
    handler = fn;
  } } });
  let body;
  handler({}, { setHeader() {}, end(value) { body = value; } });
  return JSON.parse(body);
}

test('real GPU mode has no proxy or source when it is not configured', () => {
  const config = createDashboardConfig({ PROM_URL: 'http://sim-prom:9090' });
  assert.equal(config.server.proxy['/api/prom'].target, 'http://sim-prom:9090');
  assert.equal(config.server.proxy['/api/real-prom'], undefined);
  assert.deepEqual(exposedConfig(config), { realPrometheusConfigured: false, agentConfigured: false });
});

test('real GPU mode uses its own configured Prometheus proxy', () => {
  const config = createDashboardConfig({ PROM_URL: 'http://sim-prom:9090', REAL_PROM_URL: 'http://real-prom:9090' });
  const proxy = config.server.proxy['/api/real-prom'];
  assert.equal(proxy.target, 'http://real-prom:9090');
  assert.equal(proxy.rewrite('/api/real-prom/api/v1/query'), '/api/v1/query');
  assert.equal(config.server.proxy['/api/prom'].target, 'http://sim-prom:9090');
  assert.deepEqual(exposedConfig(config), { realPrometheusConfigured: true, agentConfigured: false });
});


test('agent proxy is enabled only when its URL is configured', () => {
  const absent = createDashboardConfig({});
  assert.equal(absent.server.proxy['/api/agent'], undefined);
  assert.equal(exposedConfig(absent).agentConfigured, false);
  const configured = createDashboardConfig({ AGENT_URL: 'http://agent:8090' });
  assert.equal(configured.server.proxy['/api/agent'].target, 'http://agent:8090');
  assert.equal(configured.server.proxy['/api/agent'].rewrite('/api/agent/approvals'), '/approvals');
  assert.equal(exposedConfig(configured).agentConfigured, true);
});
