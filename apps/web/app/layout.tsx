import "./globals.css";
import Sidebar from "../components/Sidebar";

export const metadata = {
  title: "Conversational Portfolio",
  description: "Phase 1",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh">
      <body>
        <div style={{ display: "flex" }}>
          <Sidebar />
          <div style={{ flex: 1 }}>{children}</div>
        </div>
      </body>
    </html>
  );
}
