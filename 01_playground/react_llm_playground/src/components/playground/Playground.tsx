import React from "react";
import { FullScreen } from "../system/layouts/FullScreen";
import InteractivePlayground from "./InteractivePlayground";
import PlaygroundSettings from "./PlaygroundSettings";
import {
  XMarkIcon,
  AdjustmentsHorizontalIcon,
} from "@heroicons/react/20/solid";
import { Popover } from "@headlessui/react";
import ModelSelector from "./ModelSelector";

interface PlaygroundProps {
  token: string;
}

const models = [
  { name: "Mistral 7B", value: "mistralai/Mistral-7B-Instruct-v0.1" },
  { name: "Intel Neural Chat 7B", value: "Intel/neural-chat-7b-v3-1" },
  { name: "Meta Llama-2 7B", value: "meta-llama/Llama-2-7b-chat-hf" },
  { name: "Meta Llama-2 70B", value: "meta-llama/Llama-2-70b-chat-hf" },
  // { name: "test model", value: "hf-internal-testing/tiny-random-gpt2" },
  // Add more models as needed - if you add a new model, make sure to add it to the FetchQuestionAnswer.ts file as well
];

/*
 * Primary UI component for user interaction
 */
export const Playground = (props: PlaygroundProps) => {
  const [selectedModel, setSelectedModel] = React.useState(models[0]);
  const [outputLength, setOutputLength] = React.useState(2048); // Default value for output length

  return (
    <FullScreen>
      <div className="flex flex-col md:grid md:grid-cols-[auto_300px] md:gap-4 p-2 pt-14 md:p-6 md:pt-16 w-full h-full">
        <div className="md:hidden flex mb-2">
          <div className="flex-grow">
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
          </div>
          <Popover className="relative">
            {({ open }) => (
              <>
                <Popover.Button className="px-4 py-3 w-full justify-between text-sm font-semibold text-white">
                  {open ? (
                    <XMarkIcon
                      className="h-5 w-5 text-white"
                      aria-hidden="true"
                    />
                  ) : (
                    <AdjustmentsHorizontalIcon
                      className="h-5 w-5 text-white"
                      aria-hidden="true"
                    />
                  )}
                </Popover.Button>

                <Popover.Panel className="absolute h-[90vh] right-[-6px] p-[6px] z-10 mt-2 w-[100vw] max-w-[900px] origin-top-right rounded-md bg-[#121212] shadow-lg ring-1 ring-white ring-opacity-5 focus:outline-none">
                  <PlaygroundSettings
                    models={models}
                    selectedModel={selectedModel}
                    setSelectedModel={setSelectedModel}
                    outputLength={outputLength}
                    setOutputLength={setOutputLength}
                  />
                </Popover.Panel>
              </>
            )}
          </Popover>
        </div>
        <div className="grow flex flex-col h-full overflow-hidden">
          <InteractivePlayground
            token={props.token}
            default={selectedModel.value}
            custom_name="llm"
            prompt_key="llama2"
            output_len={outputLength}
            shouldFetchModel={false}
          />
        </div>
        <div className="hidden md:block relative">
          <PlaygroundSettings
            models={models}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            outputLength={outputLength}
            setOutputLength={setOutputLength}
          />
        </div>
      </div>
    </FullScreen>
  );
};
