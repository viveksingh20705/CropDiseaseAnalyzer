import axios from "axios";
import { Eye, EyeOff, ImagePlus, LogOut, Save, ShieldCheck, SunMoon, UserRound } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import { useTheme } from "../context/ThemeContext.jsx";

export default function Settings() {
  const { user, updateUser, logout } = useAuth();
  const { theme, setTheme } = useTheme();
  const navigate = useNavigate();
  const avatarInput = useRef(null);
  const [profile, setProfile] = useState({ name: user?.name || "", email: user?.email || "" });
  const [password, setPassword] = useState({ current_password: "", new_password: "", confirm_password: "" });
  const [showPasswords, setShowPasswords] = useState(false);
  const [profileMessage, setProfileMessage] = useState("");
  const [passwordMessage, setPasswordMessage] = useState("");
  const [profileError, setProfileError] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);

  const saveProfile = async (event) => {
    event.preventDefault();
    setProfileMessage("");
    setProfileError("");
    setSavingProfile(true);
    try {
      const { data } = await axios.patch("/api/auth/profile", {
        name: profile.name.trim(),
        email: profile.email.trim(),
      });
      updateUser(data);
      setProfileMessage("Your profile has been updated.");
    } catch (error) {
      setProfileError(error.response?.data?.detail || "Could not save your profile.");
    } finally {
      setSavingProfile(false);
    }
  };

  const changePassword = async (event) => {
    event.preventDefault();
    setPasswordMessage("");
    setPasswordError("");
    if (password.new_password !== password.confirm_password) {
      setPasswordError("The new passwords do not match.");
      return;
    }
    setSavingPassword(true);
    try {
      await axios.post("/api/auth/password", {
        current_password: password.current_password,
        new_password: password.new_password,
      });
      setPassword({ current_password: "", new_password: "", confirm_password: "" });
      setPasswordMessage("Your password has been changed.");
    } catch (error) {
      setPasswordError(error.response?.data?.detail || "Could not change your password.");
    } finally {
      setSavingPassword(false);
    }
  };

  const uploadAvatar = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setProfileMessage("");
    setProfileError("");
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setProfileError("Choose a JPEG, PNG or WebP image.");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    try {
      const { data } = await axios.post("/api/auth/avatar", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      updateUser(data);
      setProfileMessage("Your profile picture has been updated.");
    } catch (error) {
      setProfileError(error.response?.data?.detail || "Could not update your profile picture.");
    }
  };

  const signOut = () => {
    if (!window.confirm("Sign out of Crop Disease Analyzer?")) return;
    logout();
    navigate("/login", { replace: true });
  };

  return (
    <div className="page-stack settings-page">
      <div className="page-heading-row">
        <div><p className="eyebrow">YOUR ACCOUNT</p><h1>Settings</h1><p className="page-subtitle">Manage your profile, security and appearance.</p></div>
      </div>

      <section className="settings-card panel">
        <div className="settings-section-heading"><span className="settings-icon"><UserRound size={18} /></span><div><h2>Profile information</h2><p>Update your account details and profile picture.</p></div></div>
        <div className="profile-edit-row">
          <div className="settings-avatar">
            {user?.avatar_url ? <img src={`${axios.defaults.baseURL}${user.avatar_url}`} alt="Profile" /> : <span>{user?.name?.trim()?.[0]?.toUpperCase() || "U"}</span>}
          </div>
          <div><strong>{user?.name}</strong><span>{user?.email}</span></div>
          <button type="button" className="button button-secondary avatar-upload" onClick={() => avatarInput.current?.click()}><ImagePlus size={16} /> Change photo</button>
          <input ref={avatarInput} type="file" accept="image/png,image/jpeg,image/webp" className="sr-only" onChange={uploadAvatar} />
        </div>
        <form className="settings-form" onSubmit={saveProfile}>
          <label>Full name<input required minLength={2} maxLength={80} value={profile.name} onChange={(event) => setProfile({ ...profile, name: event.target.value })} /></label>
          <label>Email address<input required type="email" value={profile.email} onChange={(event) => setProfile({ ...profile, email: event.target.value })} /></label>
          {profileError && <p className="form-message form-error">{profileError}</p>}
          {profileMessage && <p className="form-message form-success">{profileMessage}</p>}
          <div className="settings-actions"><span className="settings-note">{user?.role === "admin" ? "Administrator account" : "Farmer account"}</span><button type="submit" className="button button-primary" disabled={savingProfile}><Save size={16} />{savingProfile ? "Saving…" : "Save profile"}</button></div>
        </form>
      </section>

      <section className="settings-card panel">
        <div className="settings-section-heading"><span className="settings-icon settings-icon-blue"><ShieldCheck size={18} /></span><div><h2>Change password</h2><p>Use your current password to set a new one.</p></div></div>
        <form className="settings-form password-form" onSubmit={changePassword}>
          {[
            ["current_password", "Current password"],
            ["new_password", "New password"],
            ["confirm_password", "Confirm new password"],
          ].map(([key, label]) => (
            <label key={key}>{label}<span className="password-field"><input required type={showPasswords ? "text" : "password"} minLength={key === "current_password" ? undefined : 8} maxLength={128} value={password[key]} onChange={(event) => setPassword({ ...password, [key]: event.target.value })} /><button type="button" onClick={() => setShowPasswords((value) => !value)} aria-label={showPasswords ? "Hide passwords" : "Show passwords"}>{showPasswords ? <EyeOff size={17} /> : <Eye size={17} />}</button></span></label>
          ))}
          {passwordError && <p className="form-message form-error">{passwordError}</p>}
          {passwordMessage && <p className="form-message form-success">{passwordMessage}</p>}
          <div className="settings-actions"><span className="settings-note">At least 8 characters</span><button type="submit" className="button button-secondary" disabled={savingPassword}>{savingPassword ? "Updating…" : "Update password"}</button></div>
        </form>
      </section>

      <section className="settings-card panel">
        <div className="settings-section-heading"><span className="settings-icon settings-icon-amber"><SunMoon size={18} /></span><div><h2>Appearance</h2><p>Choose your preferred application theme.</p></div></div>
        <div className="theme-options">
          {[
            ["dark", "Dark theme", "Easy on the eyes"],
            ["light", "White theme", "Bright and minimal"],
          ].map(([value, title, caption]) => (
            <button type="button" key={value} className={`theme-option ${theme === value ? "selected" : ""}`} onClick={() => setTheme(value)} aria-pressed={theme === value}>
              <span className={`theme-swatch ${value}`}><span /><span /><span /></span>
              <span><strong>{title}</strong><small>{caption}</small></span>
              <span className="theme-radio" />
            </button>
          ))}
        </div>
      </section>

      <section className="settings-logout panel">
        <div><h2>Sign out</h2><p>End your current session on this device.</p></div>
        <button type="button" className="button button-danger" onClick={signOut}><LogOut size={16} /> Sign out</button>
      </section>
    </div>
  );
}
