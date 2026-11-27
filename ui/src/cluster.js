const $ = selector => document.querySelector(selector);
const text = value => value === null || value === undefined || value === '' ? '—' : String(value);

export function hpaQuantity(value) {
  if (typeof value === 'string' && /^-?\d+(?:\.\d+)?m$/.test(value)) return String(Number(value.slice(0, -1)) / 1000);
  return text(value);
}

export function nodeGpuRequests(data, node) {
  if (data.errors?.allocation_pods || !Array.isArray(data.allocation_pods)) return null;
  return data.allocation_pods.filter(p => p.node === node && !['Succeeded', 'Failed'].includes(p.phase)).reduce((sum, p) => sum + p.gpu_requests, 0);
}

export function clusterSummary(data) {
  const nodeData = !data.errors?.nodes, podData = !data.errors?.pods;
  const workers = data.pods.filter(p => p.app === 'dynamo-inference-workers');
  const virtualNodes = data.nodes.filter(n => n.virtual);
  const capacity = virtualNodes.reduce((sum, n) => sum + Number(n.allocatable?.['nvidia.com/gpu'] || 0), 0);
  const allocated = workers.filter(p => p.node && !['Succeeded', 'Failed'].includes(p.phase)).reduce((sum, p) => sum + p.gpu_requests, 0);
  return { virtualNodes: nodeData ? virtualNodes.length : null, readyNodes: nodeData ? virtualNodes.filter(n => n.ready === 'True').length : null, capacity: nodeData ? capacity : null, allocated: podData ? allocated : null, running: podData ? workers.filter(p => p.phase === 'Running').length : null, pending: podData ? workers.filter(p => p.phase === 'Pending').length : null };
}

function table(container, columns, rows) {
  container.replaceChildren();
  if (!rows.length) { container.textContent = 'No data available'; return; }
  const wrap = document.createElement('div'); wrap.className = 'cluster-table-wrap';
  const table = document.createElement('table'), head = table.createTHead().insertRow();
  for (const [label] of columns) { const th = document.createElement('th'); th.textContent = label; head.append(th); }
  const body = table.createTBody();
  for (const item of rows) { const row = body.insertRow(); for (const [, value] of columns) row.insertCell().textContent = text(value(item)); }
  wrap.append(table); container.append(wrap);
}

function conditions(items) {
  return items.map(c => `${c.type}: ${c.status}${c.reason ? ` (${c.reason})` : ''}${c.message ? ` — ${c.message}` : ''}`).join('\n') || 'No conditions reported';
}

function render(data) {
  const summary = clusterSummary(data), box = $('#cluster-summary'); box.replaceChildren();
  for (const [label, value] of [['Ready virtual nodes', summary.readyNodes === null ? 'Unavailable' : `${summary.readyNodes}/${summary.virtualNodes}`], ['Advertised GPUs', summary.capacity], ['Worker GPUs requested', summary.allocated], ['Virtual workers running', summary.running], ['Workers pending', summary.pending]]) {
    const card = document.createElement('div'), strong = document.createElement('strong'), caption = document.createElement('span');
    card.className = 'cluster-stat'; strong.textContent = value === null ? 'Unavailable' : value; caption.textContent = label; card.append(strong, caption); box.append(card);
  }
  table($('#cluster-hpa'), [['HPA', h => h.name], ['Current / desired replicas', h => `${text(h.current)} / ${text(h.desired)}`], ['Min / max replicas', h => `${text(h.min)} / ${text(h.max)}`], ['Current / target metric', h => h.metrics.map((m, i) => `${text(m.external?.metric?.name)}: ${hpaQuantity(h.current_metrics[i]?.external?.current?.averageValue ?? h.current_metrics[i]?.external?.current?.value)} / ${hpaQuantity(m.external?.target?.averageValue ?? m.external?.target?.value)}`).join('\n')], ['Conditions', h => conditions(h.conditions)]], data.hpas);
  table($('#cluster-keda'), [['ScaledObject', k => k.name], ['Min / max', k => `${text(k.min)} / ${text(k.max)}`], ['Trigger', k => k.triggers.map(t => `${t.type}: ${text(t.threshold)}`).join(', ')], ['Poll / cooldown (s)', k => `${text(k.polling_interval)} / ${text(k.cooldown_period)}`], ['Conditions', k => conditions(k.conditions)], ['Last active', k => k.last_active_time]], data.scaledobjects);
  table($('#cluster-components'), [['Deployment', d => d.name], ['Desired', d => d.desired], ['Ready', d => d.ready], ['Available', d => d.available], ['Conditions', d => conditions(d.conditions)]], [...data.deployments, ...data.keda_deployments]);
  table($('#cluster-control-pods'), [['Control-plane pod', p => p.name], ['Node', p => p.node], ['Phase', p => p.phase], ['Ready', p => p.conditions.find(c => c.type === 'Ready')?.status]], data.control_plane_pods);
  table($('#cluster-nodes'), [['Node', n => n.name], ['Kind', n => n.virtual ? 'KWOK virtual' : 'Control plane'], ['Ready', n => n.ready], ['Schedulable', n => n.unschedulable ? 'No (cordoned)' : 'Yes'], ['GPUs requested / allocatable (all namespaces)', n => `${nodeGpuRequests(data, n.name) ?? 'Unavailable'} / ${text(n.allocatable?.['nvidia.com/gpu'])}`]], data.nodes);
  table($('#cluster-pods'), [['Pod', p => p.name], ['Role', p => p.app === 'dynamo-inference-workers' ? 'Virtual worker' : 'Simulator engine'], ['Node', p => p.node], ['Status', p => p.phase], ['GPUs requested', p => p.gpu_requests], ['Reason', p => p.reason || p.conditions.filter(c => c.status === 'False').map(c => c.message || c.reason).join('; ')]], data.pods);
  table($('#cluster-events'), [['Time', e => e.last_seen], ['Type', e => e.type], ['Object', e => e.object], ['Reason', e => e.reason], ['Message', e => e.message], ['Count', e => e.count]], data.events);
  const errors = Object.entries(data.errors || {});
  $('#cluster-status').textContent = errors.length ? `Partial or unavailable cluster data: ${errors.map(([name, error]) => `${name}: ${error}`).join('; ')}` : `Connected · updated ${new Date().toLocaleTimeString()} · fixed inventory; KEDA worker scaling does not change simulator capacity`;
  $('#cluster-status').className = errors.length ? 'cluster-error' : 'cluster-note';
}

export function createClusterView({ url, enabled }) {
  let sequence = 0;
  return { async refresh() {
    const version = ++sequence;
    if (!enabled()) { $('#cluster-status').textContent = 'Kubernetes data is available in simulated mode when the simulator runs inside the cluster.'; for (const id of ['cluster-summary', 'cluster-hpa', 'cluster-keda', 'cluster-components', 'cluster-control-pods', 'cluster-nodes', 'cluster-pods', 'cluster-events']) $(`#${id}`).replaceChildren(); return; }
    try {
      const response = await fetch(url());
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (version === sequence && enabled()) render(data);
    } catch (error) {
      if (version === sequence && enabled()) {
        $('#cluster-status').textContent = `Cluster observability unavailable: ${error.message}. The simulator requires the updated in-cluster image and read-only service account.`;
        $('#cluster-status').className = 'cluster-error';
        for (const id of ['cluster-summary', 'cluster-hpa', 'cluster-keda', 'cluster-components', 'cluster-control-pods', 'cluster-nodes', 'cluster-pods', 'cluster-events']) $(`#${id}`).replaceChildren();
      }
    }
  } };
}

export function renderServing(data, message = '') {
  const container = $('#cluster-serving');
  container.replaceChildren();
  $('#cluster-serving-status').textContent = message || data?.reason ||
    'KWOK predictor placement and readiness are simulated.';
  if (!data) return;

  const sections = [
    ['InferenceServices', 'services', [['Service', row => row.name], ['Reported URL', row => row.status?.url]]],
    ['Predictor deployments', 'deployments', [['Deployment', row => row.name], ['Ready / desired', row => `${row.status?.readyReplicas ?? 0} / ${text(row.spec?.replicas)}`]]],
    ['ClusterQueues', 'cluster_queues', [
      ['Queue', row => row.name],
      ['Quota by flavor', row => row.spec?.resourceGroups?.flatMap(group => group.flavors.map(flavor =>
        `${flavor.name}: ${flavor.resources.map(resource => `${resource.name} ${resource.nominalQuota}`).join(', ')}`)).join('\n')],
      ['Reserved', row => row.status?.flavorsReservation?.map(flavor =>
        `${flavor.name}: ${flavor.resources.map(resource => `${resource.name} ${resource.total}`).join(', ')}`).join('\n')],
      ['Admitted / pending', queueCounts],
    ]],
    ['LocalQueues', 'queues', [['Queue', row => row.name], ['ClusterQueue', row => row.spec?.clusterQueue], ['Admitted / pending', queueCounts]]],
    ['Workload admission', 'workloads', [['Workload', row => row.name], ['Owner Pod', row => row.owners?.find(owner => owner.kind === 'Pod')?.name], ['Queue', row => row.queue], ['Reserved in', row => row.status?.admission?.clusterQueue]]],
    ['Predictor Pod placement', 'pods', [['Pod', row => row.name], ['InferenceService', row => row.service], ['Scheduler', row => row.scheduler], ['Node', row => row.node], ['Phase', row => row.status?.phase], ['GPUs by container', row => row.gpu_requests?.join(', ')], ['Scheduling gates', row => row.gates?.map(gate => gate.name).join(', ')]]],
    ['Gateways', 'gateways', [['Gateway', row => row.name], ['Addresses', row => row.status?.addresses?.map(address => address.value).join(', ')], ['Listeners', row => row.spec?.listeners?.map(listener => `${listener.protocol}:${listener.port}`).join(', ')], ['Attached routes', row => row.status?.listeners?.map(listener => `${listener.name}: ${listener.attachedRoutes}`).join(', ')]]],
    ['HTTPRoutes', 'routes', [['Route', row => row.name], ['Hosts', row => row.spec?.hostnames?.join(', ')], ['Parent conditions', row => row.status?.parents?.map(parent => `${parent.parentRef?.namespace ?? data.namespace}/${parent.parentRef?.name}\n${conditions(parent.conditions ?? [])}`).join('\n')]]],
  ];
  for (const [title, key, columns] of sections) {
    const heading = document.createElement('h3');
    heading.textContent = title;
    const body = document.createElement('div');
    container.append(heading, body);
    const rows = data[key];
    if (!Array.isArray(rows)) {
      body.textContent = rows?.error || 'Unavailable';
      continue;
    }
    table(body, [...columns, ['Conditions', row => conditions(row.conditions ?? [])]], rows);
  }
}

function queueCounts(row) {
  return `${text(row.status?.admittedWorkloads)} / ${text(row.status?.pendingWorkloads)}`;
}
