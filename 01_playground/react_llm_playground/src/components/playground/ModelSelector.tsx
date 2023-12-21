import React from "react";
import { Menu, Transition } from "@headlessui/react";
import { Fragment } from "react";
import { ChevronDownIcon } from "@heroicons/react/20/solid";

interface ModelSelectorProps {
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
}

/*
 * Primary UI component for user interaction
 */
export default function ModelSelector({
  models,
  selectedModel,
  setSelectedModel,
}: ModelSelectorProps) {
  return (
    <Menu as="div" className="relative inline-block text-left w-full">
      <Menu.Button className="bg-[#1d1d1d] inline-flex px-4 py-3 w-full justify-between gap-x-1.5 rounded-[12px] border-[1px] border-[#393939] text-sm font-semibold text-white shadow-sm">
        <p>{selectedModel.name}</p>
        <ChevronDownIcon
          className="-mr-1 h-5 w-5 text-white"
          aria-hidden="true"
        />
      </Menu.Button>
      <Transition
        as={Fragment}
        enter="transition ease-out duration-100"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items className="absolute left-0 z-10 mt-2 w-full origin-top-left rounded-[12px] bg-[#1d1d1d] shadow-lg ring-1 ring-white ring-opacity-5 focus:outline-none">
          <div className="p-1">
            {models.map((model) => (
              <Menu.Item key={model.value}>
                <p
                  className="p-2 pl-3 hover:text-[#f1f1f1] hover:bg-[rgba(255,108,113,0.8)] cursor-pointer"
                  onClick={() => setSelectedModel(model)}
                >
                  {model.name}
                </p>
              </Menu.Item>
            ))}
          </div>
        </Menu.Items>
      </Transition>
    </Menu>
  );
}
