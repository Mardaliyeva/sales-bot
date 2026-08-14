import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Satış Köməkçisi",
  description: "Məhsul seçimi, qiymət və stok məlumatları üçün ağıllı satış köməkçisi",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="az">
      <body>{children}</body>
    </html>
  );
}
