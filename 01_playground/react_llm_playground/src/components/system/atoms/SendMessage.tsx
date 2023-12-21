import React from "react";

interface SendMessageProps {
  message: string;
  setMessage: (message: string) => void;
  onSendMessage: () => void;
  userTurn: boolean;
}

export const SendMessage = (props: SendMessageProps) => {
  const handleSubmit = (event: React.SyntheticEvent) => {
    event.preventDefault();
    if (props.message !== "") {
      props.onSendMessage();
      props.setMessage("");
    }
  };
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (props.userTurn) {
          handleSubmit(event);
        }
      }}
      className="flex w-full rounded-[12px] border-[1px] border-[#9E9E9E]"
    >
      <input
        value={props.message}
        placeholder="Type your message here…"
        className="w-full rounded-l-[12px] bg-[#1d1d1d] border-none text-white"
        onChange={(event) => props.setMessage(event.target.value)}
        onKeyDown={(event) => {
          if (
            props.userTurn &&
            event.key == "Enter" &&
            event.shiftKey == false
          ) {
            // only send message if enter is pressed without shift
            handleSubmit(event);
          }
        }}
      />
      <button className="px-4 rounded-r-[12px] bg-[#1d1d1d] border-none ml-[1px]">
        Send
      </button>
    </form>
  );
};
