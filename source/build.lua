--[==========================================[--
                BUILDING MOMENTS
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
  "03-parameter_regimes/05-lowT",
  "03-parameter_regimes/06-highT",
  "03-parameter_regimes/07-weak_coupling",
  "03-parameter_regimes/08-strong_coupling"
}

function run(dir, cmd)
  return os.execute("cd " .. dir .. ";" .. cmd)
end

for i, path in ipairs(moments_list) do
  print(string.format("[%d/%d] Calculating Moments", i, #moments_list))
  local _path = path .. "/moments/"
  os.execute("cd " .. _path .. "; python main.py")
end