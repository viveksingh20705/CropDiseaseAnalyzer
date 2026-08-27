import axios from "axios";
import { useEffect, useState } from "react";

export default function Admin() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/api/admin/stats")
      .then(({ data }) => setStats(data))
      .catch(() => setError("Admin endpoint is restricted to administrator accounts."));
  }, []);

  if (error) {
    return <p className="mx-auto max-w-6xl px-4 py-10 text-rose-700">{error}</p>;
  }
  if (!stats) {
    return <p className="mx-auto max-w-6xl px-4 py-10 text-slate-500">Loading administration stats...</p>;
  }

  const cards = [
    ["Users", stats.users],
    ["Farmers", stats.farmers],
    ["Admins", stats.admins],
    ["Predictions", stats.predictions],
    ["Healthy", stats.healthy],
    ["Diseased", stats.diseased],
    ["Crops", stats.crops],
    ["Diseases", stats.diseases],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-3xl font-bold text-leaf-900">Administration</h1>
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map(([label, value]) => (
          <article key={label} className="card p-4">
            <p className="text-sm text-slate-500">{label}</p>
            <p className="text-2xl font-bold">{value}</p>
          </article>
        ))}
      </div>
      <section className="card mt-6 p-5">
        <h2 className="font-semibold">Recent analyses</h2>
        <ul className="mt-3 divide-y divide-slate-100">
          {stats.recent_predictions.map((item) => (
            <li key={item.id} className="flex justify-between py-2 text-sm">
              <span>
                {item.crop} · {item.disease}
              </span>
              <span className="text-slate-500">{item.confidence}%</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
