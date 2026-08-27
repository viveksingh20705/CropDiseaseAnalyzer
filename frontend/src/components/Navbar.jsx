import { Leaf, Menu, X } from "lucide-react";
import { useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const linkClass = ({ isActive }) =>
  `rounded-lg px-3 py-2 text-sm font-medium transition ${
    isActive ? "bg-leaf-600 text-white" : "text-slate-700 hover:bg-leaf-50"
  }`;

export default function Navbar() {
  const { user, logout, isAdmin } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

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
        <Link to="/" className="flex items-center gap-2 font-semibold text-leaf-700">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-leaf-600 text-white">
            <Leaf size={18} />
          </span>
          Crop Disease Analyzer
        </Link>
        <nav className="hidden items-center gap-1 md:flex">
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
        </nav>
        <button
          type="button"
          className="rounded-lg p-2 text-slate-700 md:hidden"
          onClick={() => setOpen((value) => !value)}
          aria-label="Toggle menu"
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>
      {open && (
        <div className="space-y-1 border-t border-slate-100 px-4 py-3 md:hidden">
          {user &&
            links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={linkClass}
                end={link.to === "/"}
                onClick={() => setOpen(false)}
              >
                {link.label}
              </NavLink>
            ))}
          {user ? (
            <button type="button" onClick={onLogout} className="block w-full rounded-lg px-3 py-2 text-left text-sm text-rose-700">
              Logout
            </button>
          ) : (
            <NavLink to="/login" className={linkClass} onClick={() => setOpen(false)}>
              Login
            </NavLink>
          )}
        </div>
      )}
    </header>
  );
}
