import { existsSync } from "node:fs";
import net from "node:net";
import { resolve } from "node:path";
import { spawn } from "node:child_process";

const port = Number(process.env.FRONTEND_PORT || process.env.PORT || "3000");
const host = "127.0.0.1";
const lockPath = resolve(process.cwd(), ".next", "dev", "lock");

function listenerExists() {
  return new Promise((resolveCheck) => {
    const socket = net.createConnection({ host, port });
    socket.setTimeout(300);
    socket.once("connect", () => {
      socket.destroy();
      resolveCheck(true);
    });
    socket.once("timeout", () => {
      socket.destroy();
      resolveCheck(false);
    });
    socket.once("error", () => resolveCheck(false));
  });
}

if ((await listenerExists()) || existsSync(lockPath)) {
  console.log(`Next dev artıq port ${port} və ya ${lockPath} lock-u ilə işləyir; ikinci instansiya açılmadı.`);
  process.exit(0);
}

const nextBin = resolve(process.cwd(), "node_modules", "next", "dist", "bin", "next");
const child = spawn(process.execPath, [nextBin, "dev", "-p", String(port)], {
  cwd: process.cwd(),
  env: process.env,
  stdio: "inherit",
});
child.once("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 1);
});
