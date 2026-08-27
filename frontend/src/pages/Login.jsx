import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function Login() {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const onChange = (event) => {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(form.email, form.password);
      } else {
        await register(form.name, form.email, form.password);
      }
      navigate("/analyze");
    } catch (err) {
      setError(err.response?.data?.detail || "Authentication failed. Check your credentials.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto grid min-h-[70vh] max-w-5xl items-center gap-10 px-4 py-10 md:grid-cols-2">
      <div>
        <p className="text-sm font-semibold uppercase tracking-wide text-leaf-600">Field diagnostics</p>
        <h1 className="mt-2 text-4xl font-bold text-leaf-900">Identify crop diseases from a leaf photo.</h1>
        <p className="mt-4 text-slate-600">
          Upload a leaf image, receive a MobileNetV2 diagnosis, and get practical treatment and prevention steps
          for tomato, potato, corn, apple, and more.
        </p>
        <Link to="/" className="mt-6 inline-block text-sm font-medium text-leaf-700">
          Back to overview
        </Link>
      </div>
      <form onSubmit={onSubmit} className="card p-6">
        <div className="mb-4 flex rounded-xl bg-slate-100 p-1">
          {["login", "register"].map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setMode(item)}
              className={`flex-1 rounded-lg py-2 text-sm font-semibold capitalize ${
                mode === item ? "bg-white text-leaf-700 shadow" : "text-slate-500"
              }`}
            >
              {item}
            </button>
          ))}
        </div>
        {mode === "register" && (
          <label className="mb-3 block text-sm font-medium">
            Full name
            <input
              name="name"
              value={form.name}
              onChange={onChange}
              required
              className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2"
            />
          </label>
        )}
        <label className="mb-3 block text-sm font-medium">
          Email
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={onChange}
            required
            className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2"
          />
        </label>
        <label className="mb-4 block text-sm font-medium">
          Password
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={onChange}
            required
            minLength={8}
            className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2"
          />
        </label>
        {error && <p className="mb-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-xl bg-leaf-600 py-2.5 font-semibold text-white hover:bg-leaf-700 disabled:opacity-60"
        >
          {submitting ? "Please wait..." : mode === "login" ? "Sign in" : "Create account"}
        </button>
        <p className="mt-3 text-xs text-slate-500">Demo admin: admin@cropanalyzer.com / Admin@123456</p>
      </form>
    </div>
  );
}
