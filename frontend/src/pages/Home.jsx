import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function Home() {
  const { user } = useAuth();
  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <section className="overflow-hidden rounded-3xl bg-gradient-to-br from-leaf-700 to-leaf-900 p-8 text-white md:p-12">
        <p className="text-sm uppercase tracking-[0.2em] text-leaf-100">Precision plant pathology</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-bold md:text-5xl">
          Diagnose crop diseases in the field and keep a research-grade history.
        </h1>
        <p className="mt-4 max-w-2xl text-leaf-50">
          Crop Disease Analyzer combines transfer learning (MobileNetV2), a FastAPI service, and MongoDB so
          students can demonstrate a complete AI/ML product: ingest, infer, explain, and visualize.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link to={user ? "/analyze" : "/login"} className="rounded-xl bg-white !text-green-800 px-5 py-2.5 font-semibold">
            Start analysis
          </Link>
          <Link to="/dashboard" className="rounded-xl border border-white/40 px-5 py-2.5 font-semibold">
            View dashboard
          </Link>
        </div>
      </section>
      <section className="mt-10 grid gap-4 md:grid-cols-3">
        {[
          ["1. Train", "Place PlantVillage-style folders in ml/dataset and run python train.py."],
          ["2. Seed", "Load admin, crops, and the disease library into MongoDB."],
          ["3. Diagnose", "Sign in, upload a leaf, and review confidence, symptoms, and treatment."],
        ].map(([title, body]) => (
          <article key={title} className="card p-5">
            <h2 className="font-semibold text-leaf-800">{title}</h2>
            <p className="mt-2 text-sm text-slate-600">{body}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
