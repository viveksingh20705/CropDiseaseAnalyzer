import { Navigate, Outlet, Route, Routes, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import { useAuth } from "./context/AuthContext.jsx";
import DashboardLayout from "./components/DashboardLayout.jsx";
import Admin from "./pages/Admin.jsx";
import Analyze from "./pages/Analyze.jsx";
import AskAI from "./pages/AskAI.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import History from "./pages/History.jsx";
import Home from "./pages/Home.jsx";
import { About, Help } from "./pages/Information.jsx";
import Library from "./pages/Library.jsx";
import Login from "./pages/Login.jsx";
import Settings from "./pages/Settings.jsx";
import Weather from "./pages/Weather.jsx";

function ProtectedRoute({ children, adminOnly = false }) {
  const { user, loading, isAdmin } = useAuth();
  if (loading) {
    return <p className="px-4 py-10 text-center text-slate-500">Loading session...</p>;
  }
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  if (adminOnly && !isAdmin) {
    return <Navigate to="/dashboard" replace />;
  }
  return children || <Outlet />;
}

export default function App() {
  const { user } = useAuth();
  const { pathname } = useLocation();
  const showNavbar = !user && pathname !== "/" && pathname !== "/login";
  return (
    <div className="app-root">
      {showNavbar && <Navbar />}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route element={<ProtectedRoute><DashboardLayout /></ProtectedRoute>}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/history" element={<History />} />
          <Route path="/library" element={<Library />} />
          <Route path="/weather" element={<Weather />} />
          <Route path="/ask-ai" element={<AskAI />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/help" element={<Help />} />
          <Route path="/about" element={<About />} />
          <Route path="/admin" element={<ProtectedRoute adminOnly><Admin /></ProtectedRoute>} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
