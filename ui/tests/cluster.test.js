import { test } from 'node:test';
import assert from 'node:assert/strict';
import { clusterSummary, hpaQuantity, nodeGpuRequests } from '../src/cluster.js';

test('node GPU requests include serving pods and exclude terminal or unbound pods', () => {
  const data = {allocation_pods: [
    {node: 'tray-10', phase: 'Running', gpu_requests: 4},
    {node: 'tray-10', phase: 'Failed', gpu_requests: 4},
    {node: null, phase: 'Pending', gpu_requests: 4},
  ]};
  assert.equal(nodeGpuRequests(data, 'tray-10'), 4);
  assert.equal(nodeGpuRequests({errors: {allocation_pods: '403'}}, 'tray-10'), null);
  assert.equal(nodeGpuRequests({}, 'tray-10'), null);
});

test('HPA milli-quantities display as their actual metric value', () => {
  assert.equal(hpaQuantity('95389m'), '95.389');
  assert.equal(hpaQuantity('15'), '15');
  assert.equal(hpaQuantity(undefined), '—');
});

test('counts only virtual inventory and bound nonterminal worker requests', () => {
  const result = clusterSummary({errors: {}, nodes: [
    {virtual: true, ready: 'True', allocatable: {'nvidia.com/gpu': '4'}},
    {virtual: true, ready: 'False', allocatable: {'nvidia.com/gpu': '4'}},
    {virtual: false, ready: 'True', allocatable: {}},
  ], pods: [
    {app: 'dynamo-inference-workers', node: 'tray-1', phase: 'Running', gpu_requests: 4},
    {app: 'dynamo-inference-workers', node: null, phase: 'Pending', gpu_requests: 4},
    {app: 'dynamo-inference-workers', node: 'tray-2', phase: 'Failed', gpu_requests: 4},
    {app: 'gpu-simulator-engine', node: 'control', phase: 'Running', gpu_requests: 0},
  ]});
  assert.deepEqual(result, {virtualNodes: 2, readyNodes: 1, capacity: 8, allocated: 4, running: 1, pending: 1});
});

test('missing permissions do not become healthy zero counts', () => {
  const result = clusterSummary({errors: {nodes: '403', pods: '403'}, nodes: [], pods: []});
  assert.deepEqual(result, {virtualNodes: null, readyNodes: null, capacity: null, allocated: null, running: null, pending: null});
});
