/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Segoe UI", "system-ui", "sans-serif"],
      },
      colors: {
        leaf: {
          50: "#f3faf4",
          100: "#e4f3e6",
          500: "#2f8f46",
          600: "#24753a",
          700: "#1c5c2e",
          900: "#12361c",
        },
      },
    },
  },
  plugins: [],
};
