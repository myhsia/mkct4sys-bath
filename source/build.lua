--[==========================================[--
              CLACULATING MOMENTS
--]==========================================]--

local moments_list = {
  "00-spin_boson_linear",
  "01-spin_boson_quad",
  "02-spin_boson_linear_quad",
  "03-parameter_regimes/00-base_line",
  "03-parameter_regimes/01-slow_system",
  "03-parameter_regimes/02-no_bias",
  "03-parameter_regimes/03-fast_bath",
  "03-parameter_regimes/04-slow_bath",
  "03-parameter_regimes/05-lowT",
  "03-parameter_regimes/05-llowT",
  "03-parameter_regimes/06-highT",
  "03-parameter_regimes/06-hhighT",
  "03-parameter_regimes/07-weak_coupling",
  "03-parameter_regimes/08-strong_coupling"
}

local function format_time(seconds)
  local hours = math.floor(seconds / 3600)
  local mins = math.floor((seconds % 3600) / 60)
  local secs = seconds % 60
  return string.format("%02d:%02d:%02d", hours, mins, secs)
end

local start_overall = os.time()
for i, path in ipairs(moments_list) do
  print(string.format("[%d/%d] Calculating Moments", i, #moments_list))
  local _path = path .. "/moments/"
  os.execute("cd " .. _path .. "; python main.py")
  print()
end
local end_overall = os.time()

local elapsed = os.difftime(end_overall, start_overall)
print("All tasks completed.\nTotal running time: " .. format_time(elapsed))
