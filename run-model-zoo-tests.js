#!/usr/bin/env node
const { execSync } = require('child_process');
const path = require('path');

const script = path.resolve(__dirname, 'tools/model_zoo/test_generator/test_models.sh');
const datasetDir = '/usr/share/migraphx/migraph_datasets';

try {
  execSync(`${script} ${datasetDir}`, { stdio: 'inherit' });
  console.log('Model-zoo tests completed successfully.');
} catch (err) {
  console.error('Model-zoo tests failed:', err.message);
  process.exit(err.status || 1);
}
