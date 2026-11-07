import { Leaf } from "lucide-react";
import { useState } from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar.jsx";

export default function DashboardLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="dashboard-layout">
      <Sidebar
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
      />
      <div className="dashboard-main">
        <header className="mobile-topbar">
          <button
            type="button"
            className="mobile-brand"
            onClick={() => setMobileOpen((value) => !value)}
            aria-label={mobileOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={mobileOpen}
          >
            <span className="brand-mark"><Leaf size={15} /></span>
            <span>Crop Disease Analyzer</span>
          </button>
        </header>
        <main className="dashboard-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
