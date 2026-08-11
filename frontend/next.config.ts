import type { NextConfig } from "next";

const apiPort = process.env.SALES_BOT_API_PORT || "8001";
const apiBaseUrl = (
  process.env.SALES_BOT_API_URL || `http://127.0.0.1:${apiPort}`
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${apiBaseUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
