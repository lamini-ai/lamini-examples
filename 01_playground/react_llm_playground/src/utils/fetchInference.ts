import React from "react";
import { fetchStreamingCustom } from "./fetchCustom";

export default function fetchStreamingInference(
  message: string,
  modelName: string | null,
  setModelResponse: any,
  token: string,
  output_len: number
) {
  return fetchStreamingCustom(message, modelName, setModelResponse, token, output_len);
}
