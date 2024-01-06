import React from "react";

interface FullScreenProps {
  children: React.ReactNode;
}

export const FullScreen = ({ children }: FullScreenProps) => {
  return <div className="absolute inset-0">{children}</div>;
};

