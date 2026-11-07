import { Leaf, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const linkClass = ({ isActive }) =>
  `rounded-lg px-3 py-2 text-sm font-medium transition ${
    isActive ? "bg-leaf-600 text-white" : "text-slate-700 hover:bg-leaf-50"
  }`;

export default function Navbar() {
  const { user, logout, isAdmin } = useAuth();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const isMinimalHeader = pathname === "/" || pathname === "/login";

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  const links = [
    { to: "/", label: "Home" },
    { to: "/analyze", label: "Analyze" },
    { to: "/dashboard", label: "Dashboard" },
    { to: "/history", label: "History" },
    { to: "/library", label: "Disease Library" },
  ];
  if (isAdmin) {
    links.push({ to: "/admin", label: "Admin" });
  }

  const onLogout = () => {
    logout();
    setOpen(false);
    navigate("/login");
  };

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        {isMinimalHeader ? (
          <div className="navbar-brand-link flex items-center gap-2 font-semibold text-leaf-700">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-leaf-600 text-white">
              <Leaf size={18} />
            </span>
            Crop Disease Analyzer
          </div>
        ) : (
          <Link to="/" className="navbar-brand-link flex items-center gap-2 font-semibold text-leaf-700">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-leaf-600 text-white">
              <Leaf size={18} />
            </span>
            Crop Disease Analyzer
          </Link>
        )}
        {isMinimalHeader ? (
          <div className="mobile-brand">
            <span className="brand-mark"><Leaf size={16} /></span>
            <span>Crop Disease Analyzer</span>
          </div>
        ) : (
          <button
            type="button"
            className="mobile-brand"
            onClick={() => setOpen((value) => !value)}
            aria-label={open ? "Close navigation" : "Open navigation"}
            aria-expanded={open}
          >
            <span className="brand-mark"><Leaf size={16} /></span>
            <span>Crop Disease Analyzer</span>
          </button>
        )}
        {!isMinimalHeader && <nav className="navbar-links">
          {user &&
            links.map((link) => (
              <NavLink key={link.to} to={link.to} className={linkClass} end={link.to === "/"}>
                {link.label}
              </NavLink>
            ))}
          {user ? (
            <button
              type="button"
              onClick={onLogout}
              className="ml-2 rounded-lg px-3 py-2 text-sm font-medium text-rose-700 hover:bg-rose-50"
            >
              Logout
            </button>
          ) : (
            <NavLink to="/login" className={linkClass}>
              Login
            </NavLink>
          )}
        </nav>}
      </div>
      {!isMinimalHeader && <>
        <button
          type="button"
          className={`public-nav-backdrop${open ? " open" : ""}`}
          onClick={() => setOpen(false)}
          aria-label="Close navigation"
          tabIndex={open ? 0 : -1}
        />
        <nav className={`public-nav-drawer${open ? " open" : ""}`} aria-label="Mobile navigation" aria-hidden={!open}>
          <div className="public-nav-heading">
            <span>Navigation</span>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close navigation" tabIndex={open ? 0 : -1}>
              <X size={19} />
            </button>
          </div>
          {user &&
            links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={linkClass}
                end={link.to === "/"}
                onClick={() => setOpen(false)}
                tabIndex={open ? 0 : -1}
              >
                {link.label}
              </NavLink>
            ))}
          {user ? (
            <button type="button" onClick={onLogout} className="block w-full rounded-lg px-3 py-2 text-left text-sm text-rose-700" tabIndex={open ? 0 : -1}>
              Logout
            </button>
          ) : (
            <NavLink to="/login" className={linkClass} onClick={() => setOpen(false)} tabIndex={open ? 0 : -1}>
              Login
            </NavLink>
          )}
        </nav>
      </>}
    </header>
  );
}
