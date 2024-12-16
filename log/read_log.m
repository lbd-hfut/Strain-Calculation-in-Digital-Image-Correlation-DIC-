clear
clc

% 读取文本文件
filename = 'star_3_50_tanh_4.txt';

fileID = fopen(filename, 'r');

% 检查文件是否成功打开
if fileID == -1
    error('无法打开文件: %s', filename);
end

% 初始化存储结果的数组
mae_values = [];

% 按行读取文件内容
line = fgetl(fileID);
while ischar(line)
    % 使用正则表达式提取 MAE 后的值
    tokens = regexp(line, 'MAE\s+(\d+\.\d+)', 'tokens');
    if ~isempty(tokens)
        % 将提取的值转换为数字并存储
        mae_values(end+1) = str2double(tokens{1}{1});
    end
    % 读取下一行
    line = fgetl(fileID);
end
mae_values = mae_values';
% 关闭文件
fclose(fileID);

% 显示提取的值
disp('提取的 MAE 值:');
disp(mae_values);
