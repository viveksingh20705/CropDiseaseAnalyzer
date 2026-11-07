import {
  BookOpen,
  CircleHelp,
  CloudSun,
  History,
  Info,
  LayoutDashboard,
  Leaf,
  LogOut,
  MessageCircle,
  Moon,
  ScanLine,
  Settings,
  ShieldCheck,
  Sun,
} from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import { useTheme } from "../context/ThemeContext.jsx";
import axios from "axios";

const navigation = [
  { to: "/dashboard", label: "Home", icon: LayoutDashboard },
  { to: "/analyze", label: "Analyze Disease", icon: ScanLine },
  { to: "/library", label: "Disease Library", icon: BookOpen },
  { to: "/weather", label: "Weather", icon: CloudSun },
  { to: "/history", label: "History", icon: History },
  { to: "/ask-ai", label: "Ask AI", icon: MessageCircle },
  { to: "/settings", label: "Settings", icon: Settings },
];

function Avatar({ user, className = "" }) {
  const avatar = user?.avatar_url
    ? `${axios.defaults.baseURL}${user.avatar_url}`
    : "";
  return (
    <span className={`profile-avatar ${className}`}>
      {avatar ? (
        <img src={avatar} alt="" />
      ) : (
        <span aria-hidden="true">{user?.name?.trim()?.[0]?.toUpperCase() || "U"}</span>
      )}
    </span>
  );
}

export default function Sidebar({ open, onClose }) {
  const { user, logout, isAdmin } = useAuth();
  const { theme, setTheme } = useTheme();
  const navigate = useNavigate();

  const handleLogout = () => {
    if (!window.confirm("Sign out of Crop Disease Analyzer?")) return;
    logout();
    navigate("/login", { replace: true });
  };

  const linkClass = ({ isActive }) =>
    `sidebar-link${isActive ? " active" : ""}`;

  return (
    <>
      {open && <button className="sidebar-backdrop" onClick={onClose} aria-label="Close navigation" />}
      <aside className={`sidebar${open ? " mobile-open" : ""}`}>
        <div className="sidebar-brand">
          <NavLink to="/dashboard" className="brand-lockup" onClick={onClose}>
            <span className="brand-mark"><Leaf size={19} /></span>
            <span className="brand-name">Crop Disease<span>Analyzer</span></span>
          </NavLink>
        </div>

        <NavLink to="/settings" className="profile-summary" onClick={onClose}>
          <Avatar user={user} />
          <span className="profile-copy">
            <strong>{user?.name || "Your profile"}</strong>
            <span>{user?.email || user?.role || "Account settings"}</span>
          </span>
          <Settings className="profile-settings-icon" size={16} />
        </NavLink>

        <div className="sidebar-scroll">
          <p className="sidebar-section-label">Workspace</p>
          <nav className="sidebar-nav" aria-label="Main navigation">
            {navigation.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} className={linkClass} onClick={onClose} title={label}>
                <Icon size={18} strokeWidth={1.8} />
                <span>{label}</span>
              </NavLink>
            ))}
            {isAdmin && (
              <NavLink to="/admin" className={linkClass} onClick={onClose} title="Administration">
                <ShieldCheck size={18} strokeWidth={1.8} />
                <span>Administration</span>
              </NavLink>
            )}
          </nav>

          <div className="sidebar-shortcuts">
            <NavLink to="/analyze" className="sidebar-shortcut analyze-shortcut" onClick={onClose} title="Analyze Crop Image">
              <ScanLine size={17} /><span>Analyze Crop Image</span>
            </NavLink>
            <NavLink to="/ask-ai" className="sidebar-shortcut assistant-shortcut" onClick={onClose} title="AI Assistant Chat">
              <MessageCircle size={17} /><span>AI Assistant Chat</span>
            </NavLink>
          </div>
        </div>

        <div className="sidebar-footer">
          <NavLink to="/help" className="sidebar-link utility-link" onClick={onClose}>
            <CircleHelp size={18} /><span>Help &amp; Support</span>
          </NavLink>
          <NavLink to="/about" className="sidebar-link utility-link" onClick={onClose}>
            <Info size={18} /><span>About Application</span>
          </NavLink>
          <div className="sidebar-footer-row">
            <button type="button" className="sidebar-link signout-link" onClick={handleLogout}>
              <LogOut size={18} /><span>Sign Out</span>
            </button>
            <button
              type="button"
              className="theme-quick-toggle"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={`Switch to ${theme === "dark" ? "white" : "dark"} theme`}
              title={`Switch to ${theme === "dark" ? "white" : "dark"} theme`}
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
