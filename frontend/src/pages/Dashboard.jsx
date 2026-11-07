import axios from "axios";
import {
  ArrowUpRight,
  CloudSun,
  HeartPulse,
  Leaf,
  Lightbulb,
  ScanLine,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import PredictionImage from "../components/PredictionImage.jsx";

function greetingName(name = "") {
  return name.trim().split(/\s+/)[0] || "there";
}

function displayDisease(name = "") {
  return name.replaceAll("___", " · ").replaceAll("_", " ");
}

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([
      axios.get("/api/dashboard/stats"),
      axios.get("/api/predictions/history"),
    ])
      .then(([statsResponse, historyResponse]) => {
        if (!active) return;
        setStats(statsResponse.data);
        setRecent(historyResponse.data.slice(0, 4));
      })
      .catch(() => {
        if (active) setError("Dashboard information could not be loaded. Please try again.");
      });
    return () => { active = false; };
  }, []);

  const mostDetected = useMemo(
    () => stats?.disease_distribution?.find(
      (item) => item.name && !item.name.toLowerCase().includes("healthy")
    )?.name,
    [stats]
  );
  const healthRate = stats?.total_predictions
    ? Math.round((stats.healthy_count / stats.total_predictions) * 100)
    : null;
  const latestRecommendation = recent.find((item) => item.prevention?.length)?.prevention?.[0];

  if (!stats && !error) {
    return <div className="page-loading"><span className="loading-spinner" />Loading your crop overview…</div>;
  }

  return (
    <div className="page-stack">
      <div className="page-heading-row">
        <div>
          <p className="eyebrow">FIELD OVERVIEW <span className="live-dot" /> LIVE</p>
          <h1>Your farm dashboard</h1>
          <p className="page-subtitle">A clear view of your crop health and recent diagnoses.</p>
        </div>
        <Link to="/analyze" className="button button-primary"><ScanLine size={17} /> New analysis</Link>
      </div>

      {error && <div className="notice notice-error" role="alert">{error}</div>}

      <section className="welcome-banner">
        <div className="welcome-copy">
          <span className="welcome-kicker"><Sparkles size={14} /> YOUR SEASON, AT A GLANCE</span>
          <h2>Welcome back, {greetingName(user?.name)}!</h2>
          <p>Here’s your crop health overview.</p>
        </div>
        <div className="welcome-stat-grid">
          <div className="welcome-stat"><span>Total predictions</span><strong>{stats?.total_predictions ?? "—"}</strong></div>
          <div className="welcome-stat"><span>Healthy crops</span><strong>{stats?.healthy_count ?? "—"}</strong></div>
          <div className="welcome-stat"><span>Diseases detected</span><strong>{stats?.diseased_count ?? "—"}</strong></div>
        </div>
        <Leaf className="banner-leaf banner-leaf-one" size={140} />
        <Leaf className="banner-leaf banner-leaf-two" size={82} />
      </section>

      <section className="overview-grid" aria-label="Crop health summary">
        <article className="overview-card">
          <div className="overview-icon icon-green"><HeartPulse size={19} /></div>
          <div className="overview-body">
            <span className="card-overline">CROP HEALTH</span>
            <strong>{healthRate === null ? "No analyses yet" : `${healthRate}% healthy`}</strong>
            <span className="overview-caption">
              {stats?.total_predictions ? `${stats.healthy_count} healthy results from ${stats.total_predictions} analyses` : "Analyze a crop to begin tracking"}
            </span>
          </div>
          {healthRate !== null && <span className="metric-pill positive">{healthRate}%</span>}
        </article>
        <article className="overview-card">
          <div className="overview-icon icon-amber"><ShieldAlert size={19} /></div>
          <div className="overview-body">
            <span className="card-overline">DISEASE DETECTION</span>
            <strong>{stats?.diseased_count ?? 0} detected</strong>
            <span className="overview-caption">{mostDetected ? `Most frequent · ${displayDisease(mostDetected)}` : "No disease results recorded"}</span>
          </div>
          <span className="metric-pill warning">{stats?.diseased_count ?? 0}</span>
        </article>
        <Link to="/weather" className="overview-card overview-weather">
          <div className="overview-icon icon-blue"><CloudSun size={20} /></div>
          <div className="overview-body">
            <span className="card-overline">LOCAL WEATHER</span>
            <strong>Check your conditions</strong>
            <span className="overview-caption">Use your location for current weather</span>
          </div>
          <ArrowUpRight size={17} className="overview-arrow" />
        </Link>
      </section>

      <section className="dashboard-columns">
        <article className="panel recent-panel">
          <div className="panel-header">
            <div><p className="card-overline">YOUR ACTIVITY</p><h2>Recent disease analysis</h2></div>
            <Link to="/history" className="text-link">View all <ArrowUpRight size={15} /></Link>
          </div>
          {recent.length ? (
            <div className="analysis-list">
              {recent.map((item) => (
                <div className="analysis-row" key={item.id}>
                  <PredictionImage
                    src={`${axios.defaults.baseURL}${item.image_url}`}
                    alt={`${item.crop} leaf`}
                    className="analysis-thumb"
                  />
                  <div className="analysis-info">
                    <strong>{item.crop}</strong>
                    <span>{displayDisease(item.disease)}</span>
                  </div>
                  <div className="analysis-meta">
                    <span className={`status-tag ${item.is_healthy ? "status-healthy" : "status-disease"}`}>
                      {item.is_healthy ? "Healthy" : "Disease"}
                    </span>
                    <span>{item.confidence}% confidence</span>
                  </div>
                  <time>{new Date(item.created_at).toLocaleDateString()}</time>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <span className="empty-icon"><ScanLine size={21} /></span>
              <strong>No analyses yet</strong>
              <span>Your crop scans and diagnoses will show up here.</span>
              <Link to="/analyze" className="text-link">Analyze your first crop <ArrowUpRight size={15} /></Link>
            </div>
          )}
        </article>

        <article className="panel insight-panel">
          <div className="panel-header">
            <div><p className="card-overline">FIELD NOTES</p><h2>Crop health insights</h2></div>
            <span className="insight-badge"><Lightbulb size={15} /></span>
          </div>
          <div className="insight-callout">
            <span className="insight-bulb"><Lightbulb size={18} /></span>
            <div>
              <strong>{latestRecommendation ? "From your recent diagnosis" : "Build your prevention plan"}</strong>
              <p>{latestRecommendation || "Explore the disease library for prevention and treatment guidance for your crops."}</p>
            </div>
          </div>
          <div className="insight-footer">
            <span><Leaf size={16} /> Practical guidance, from your crop records</span>
            <Link to="/library" className="text-link">Disease library <ArrowUpRight size={15} /></Link>
          </div>
        </article>
      </section>

      <section className="quick-actions">
        <div><p className="card-overline">KEEP MOVING</p><h2>What would you like to do?</h2></div>
        <div className="quick-action-links">
          <Link to="/analyze" className="quick-action quick-green"><ScanLine size={18} /><span><strong>Analyze a crop</strong><small>Upload a leaf photo</small></span><ArrowUpRight size={16} /></Link>
          <Link to="/ask-ai" className="quick-action quick-blue"><Sparkles size={18} /><span><strong>Ask the crop guide</strong><small>Find advice in the library</small></span><ArrowUpRight size={16} /></Link>
          <Link to="/history" className="quick-action quick-amber"><HeartPulse size={18} /><span><strong>Review your history</strong><small>See previous diagnoses</small></span><ArrowUpRight size={16} /></Link>
        </div>
      </section>
    </div>
  );
}
