import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        pitch: "#34d399",
        line: "#12362f",
        ink: "#f8fafc",
        panel: "#0d1320",
        accent: "#22d3ee",
        soft: "#06080d",
      },
    },
  },
  plugins: [],
};

export default config;
