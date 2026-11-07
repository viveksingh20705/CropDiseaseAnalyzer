import { Link, Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import PublicTitle from "../components/PublicTitle.jsx";

export default function Home() {
  const { user } = useAuth();
  if (user) return <Navigate to="/dashboard" replace />;
  return (
    <div className="home-page mx-auto max-w-6xl px-4 py-12">
      <PublicTitle />
      <section className="home-hero overflow-hidden rounded-3xl p-8 md:p-12">
        <p className="home-kicker text-sm uppercase tracking-[0.2em]">Precision plant pathology</p>
        <h2 className="home-title mt-3 max-w-3xl text-4xl font-bold md:text-5xl">
          Diagnose crop diseases in the field and keep a research-grade history.
        </h2>
        <p className="home-description mt-4 max-w-2xl">
          Crop Disease Analyzer combines transfer learning (MobileNetV2), a FastAPI service, and MongoDB so
          students can demonstrate a complete AI/ML product: ingest, infer, explain, and visualize.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link to="/login" className="home-primary rounded-xl px-5 py-2.5 font-semibold">
            Start analysis
          </Link>
          <Link to="/dashboard" className="home-secondary rounded-xl border px-5 py-2.5 font-semibold">
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
          <article key={title} className="home-step card p-5">
            <h2 className="font-semibold">{title}</h2>
            <p className="mt-2 text-sm">{body}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
