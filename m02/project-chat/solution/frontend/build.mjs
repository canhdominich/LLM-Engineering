import { readFileSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

const out = new URL("../anhti_chat/static/", import.meta.url);
for (const name of ["sse", "app"]) {
  const ts = readFileSync(new URL(`src/${name}.ts`, import.meta.url), "utf8");
  writeFileSync(new URL(`${name}.js`, out), stripTypeScriptTypes(ts));
  console.log(`built static/${name}.js`);
}
