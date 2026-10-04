const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const ts = require("typescript");

function loadApi(fetch) {
  const source = readFileSync(path.join(__dirname, "../src/lib/api.ts"), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  });
  const context = { exports: {}, process: { env: {} }, fetch, TextDecoder };
  vm.runInNewContext(outputText, context);
  return context.exports;
}

test("a long conversation sends bounded history and the complete latest question", async () => {
  let payload;
  const api = loadApi(async (url, options) => {
    payload = JSON.parse(options.body);
    return new Response('{"type":"done","warnings":[]}\n');
  });
  const messages = Array.from({ length: 41 }, (_, index) => ({
    role: index % 2 ? "assistant" : "user", content: `Message ${index}`
  }));
  messages[39].content = "A long reply. ".repeat(1000);
  const originalReply = messages[39].content;
  await api.streamChat(messages, "punjab", () => {});
  assert.equal(payload.messages.length, 9);
  assert.equal(payload.messages[0].content, "Message 32");
  assert.deepEqual(payload.messages.at(-1), messages.at(-1));
  assert.ok(payload.messages.every((message) => Array.from(message.content).length <= 8000));
  assert.equal(payload.jurisdiction, "punjab");
  assert.equal(messages[39].content, originalReply);
});

test("shortening history preserves Unicode characters", () => {
  const api = loadApi();
  const history = api.prepareChatHistory([
    { role: "assistant", content: "\u{1f600}".repeat(9000) },
    { role: "user", content: "Continue" }
  ]);
  assert.equal(Array.from(history[0].content).length, 8000);
  assert.equal(history[0].content.includes("\uFFFD"), false);
  assert.equal(history[1].content, "Continue");
});
