import React from "react";
import { Menu, Transition } from "@headlessui/react";
import { Fragment } from "react";
import { ChevronDownIcon } from "@heroicons/react/20/solid";
import ModelSelector from "./ModelSelector";

interface PlaygroundSettingsProps {
  models: {
    name: string;
    value: string;
  }[];
  selectedModel: {
    name: string;
    value: string;
  };
  setSelectedModel: React.Dispatch<
    React.SetStateAction<{
      name: string;
      value: string;
    }>
  >;
  outputLength: number;
  setOutputLength: React.Dispatch<React.SetStateAction<number>>;
}

/*
 * Primary UI component for user interaction
 */
export default function PlaygroundSettings({
  models,
  selectedModel,
  setSelectedModel,
  outputLength,
  setOutputLength,
}: PlaygroundSettingsProps) {
  const handleSliderChange = (e) => {
    const newValue = parseInt(e.target.value, 10);
    setOutputLength(newValue);
  };

  return (
    <div>
      <div
        className="mb-4 hidden md:block border border-[#393939] rounded-[16px] p-4"
        style={{
          alignItems: "center",
        }}
      >
        <h1>Model</h1>
        <ModelSelector
          models={models}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
        />
      </div>

      <div
        className="mb-4 border border-[#393939] rounded-[16px] p-4"
        style={{
          alignItems: "center",
        }}
      >
        <h1>Output Length</h1>
        <input
          type="range"
          min="0"
          max="4000"
          step="1"
          value={outputLength}
          onChange={handleSliderChange}
          style={{
            height: 4,
            borderRadius: 4,
            outline: "none",
            WebkitAppearance: "none",
            MozAppearance: "none",
            appearance: "none",
            backgroundColor: "#9E9E9E",
          }}
          className="w-full"
        />
        <label className="pl-4 text-center" htmlFor="outputLength">
          {outputLength}
        </label>
      </div>
    </div>
  );
}
