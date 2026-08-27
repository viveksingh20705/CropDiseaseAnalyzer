import axios from "axios";
import { Activity, HeartPulse, ScanLine, ShieldAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const COLORS = ["#24753a", "#eab308", "#ea580c", "#dc2626", "#2563eb", "#7c3aed"];

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get("/api/dashboard/stats")
      .then(({ data }) => setStats(data))
      .catch(() => setError("Could not load dashboard statistics."));
  }, []);

  if (error) {
    return <p className="mx-auto max-w-6xl px-4 py-10 text-rose-700">{error}</p>;
  }
  if (!stats) {
    return <p className="mx-auto max-w-6xl px-4 py-10 text-slate-500">Loading dashboard...</p>;
  }

  const cards = [
    { label: "Total Analyses", value: stats.total_predictions, icon: ScanLine },
    { label: "Healthy", value: stats.healthy_count, icon: HeartPulse },
    { label: "Diseased", value: stats.diseased_count, icon: ShieldAlert },
    { label: "Avg Confidence", value: `${stats.avg_confidence}%`, icon: Activity },
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-3xl font-bold text-leaf-900">Dashboard</h1>
      <p className="mt-2 text-slate-600">Summary of leaf analyses stored for your account.</p>
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card) => (
          <article key={card.label} className="card p-4">
            <card.icon className="text-leaf-600" size={20} />
            <p className="mt-3 text-sm text-slate-500">{card.label}</p>
            <p className="text-2xl font-bold">{card.value}</p>
          </article>
        ))}
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <section className="card p-4">
          <h2 className="mb-4 font-semibold">Disease distribution</h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={stats.disease_distribution} dataKey="value" nameKey="name" outerRadius={90} label>
                  {stats.disease_distribution.map((entry, index) => (
                    <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="card p-4">
          <h2 className="mb-4 font-semibold">Crop breakdown</h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.crop_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" fill="#24753a" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </div>
  );
}
