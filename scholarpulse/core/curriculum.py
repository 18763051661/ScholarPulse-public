"""计算机专业核心 10 门课程的标准课程库（操作系统 + 9 门）。

每门课 30 考点 × 3 难度层(foundation/consolidation/extension) = 90 题。
共 10 门 = 300 考点 / 900 题。考点带真实先修依赖链，题目由模板确定性生成，
保证内容准确、零 LLM 依赖、可离线复现。install() 合并入 concepts.json / quiz_bank.json，
已存在的 OS 考点自动补 course 字段、去重后追加新课程。
"""
from __future__ import annotations
from ..models import Concept, QuizItem

COURSES = {
    "os": "操作系统",
    "ds": "数据结构",
    "net": "计算机网络",
    "db": "数据库系统",
    "co": "计算机组成原理",
    "algo": "算法设计与分析",
    "dm": "离散数学",
    "comp": "编译原理",
    "se": "软件工程",
    "sec": "信息安全",
}


# 每条 = (cid, name, chapter, summary, [prereq_cids], exam_weight, tier)
_DS = [
    ("ds_array", "数组随机存取", "第一章 线性结构", "连续存储、O(1)下标访问的基本线性结构", [], 0.8, "foundational"),
    ("ds_linked_list", "链表与指针", "第一章 线性结构", "通过指针链接的离散存储线性表，支持O(1)插入删除", ["ds_array"], 0.85, "foundational"),
    ("ds_stack", "栈LIFO", "第一章 线性结构", "后进先出的受限线性表，用于函数调用与表达式求值", ["ds_array"], 0.7, "foundational"),
    ("ds_queue", "队列FIFO", "第一章 线性结构", "先进先出的受限线性表，用于BFS与缓冲区调度", ["ds_array"], 0.7, "foundational"),
    ("ds_circular_queue", "循环队列", "第一章 线性结构", "取模实现首尾相接的队列，避免假溢出", ["ds_queue"], 0.6, "core"),
    ("ds_deque", "双端队列", "第一章 线性结构", "两端均可入出的线性表，单调队列的基础", ["ds_queue", "ds_stack"], 0.5, "core"),
    ("ds_hash_table", "哈希表", "第一章 线性结构", "哈希函数+冲突处理实现O(1)平均查找的字典结构", ["ds_array"], 0.9, "core"),
    ("ds_hash_collision", "哈希冲突处理", "第一章 线性结构", "开放寻址法与链地址法解决哈希冲突的两种策略", ["ds_hash_table"], 0.75, "core"),
    ("ds_tree_bst", "二叉搜索树BST", "第二章 树结构", "左<根<右的二叉树，中序遍历有序，查找O(logn)", [], 0.85, "core"),
    ("ds_avl", "AVL平衡树", "第二章 树结构", "任一节点左右子树高度差≤1的BST，LL/LR/RR/RL旋转", ["ds_tree_bst"], 0.8, "advanced"),
    ("ds_red_black", "红黑树", "第二章 树结构", "五条性质保证最长路径≤2倍最短的近似平衡BST", ["ds_tree_bst", "ds_avl"], 0.85, "advanced"),
    ("ds_b_tree", "B树", "第二章 树结构", "多路平衡查找树，磁盘IO友好的外存索引结构", ["ds_tree_bst"], 0.75, "advanced"),
    ("ds_bplus_tree", "B+树", "第二章 树结构", "B树变体，数据只在叶节点、叶节点链表，数据库索引核心", ["ds_b_tree"], 0.9, "advanced"),
    ("ds_heap", "堆", "第二章 树结构", "完全二叉树实现的优先队列，堆顶为最值", ["ds_array"], 0.8, "core"),
    ("ds_trie", "字典树Trie", "第二章 树结构", "按字符拆分的字符串前缀树，公共前缀共享存储", [], 0.65, "core"),
    ("ds_graph_rep", "图存储表示", "第三章 图结构", "邻接矩阵与邻接表两种图的存储方式", [], 0.8, "foundational"),
    ("ds_graph_dfs", "深度优先搜索DFS", "第三章 图结构", "递归沿一条路径走到底再回溯的图遍历", ["ds_graph_rep", "ds_stack"], 0.85, "core"),
    ("ds_graph_bfs", "广度优先搜索BFS", "第三章 图结构", "逐层扩展的图遍历，可求无权最短路径", ["ds_graph_rep", "ds_queue"], 0.85, "core"),
    ("ds_dijkstra", "Dijkstra最短路", "第三章 图结构", "贪心+优先队列求非负权单源最短路径", ["ds_graph_bfs", "ds_heap"], 0.9, "advanced"),
    ("ds_floyd", "Floyd全源最短路", "第三章 图结构", "动态规划求所有点对最短路径O(n^3)", ["ds_dijkstra"], 0.7, "advanced"),
    ("ds_mst_prim", "Prim最小生成树", "第三章 图结构", "从一点扩展切分最近点的MST贪心算法", ["ds_heap"], 0.75, "advanced"),
    ("ds_mst_kruskal", "Kruskal最小生成树", "第三章 图结构", "按边权排序+并查集选环外最小边的MST算法", ["ds_mst_prim"], 0.75, "advanced"),
    ("ds_topo_sort", "拓扑排序", "第三章 图结构", "对DAG节点线性化，BFS入度法或DFS后序逆序", ["ds_graph_dfs"], 0.8, "core"),
    ("ds_union_find", "并查集", "第三章 图结构", "树形结构的等价类维护，路径压缩+按秩合并近O(1)", [], 0.85, "core"),
    ("ds_sort_bubble", "冒泡/插入排序", "第四章 排序与查找", "相邻交换与逐个插入的简单O(n^2)排序", [], 0.5, "foundational"),
    ("ds_sort_quick", "快速排序", "第四章 排序与查找", "分治选取基准划分区间，平均O(nlogn)原地排序", ["ds_sort_bubble"], 0.9, "core"),
    ("ds_sort_merge", "归并排序", "第四章 排序与查找", "分治+合并有序子列的稳定O(nlogn)排序", ["ds_sort_bubble"], 0.85, "core"),
    ("ds_sort_heap", "堆排序", "第四章 排序与查找", "建堆+逐次取堆顶的原地O(nlogn)排序", ["ds_heap"], 0.75, "core"),
    ("ds_sort_radix", "基数/计数排序", "第四章 排序与查找", "非比较的线性时间排序，依赖键值范围有限", ["ds_sort_bubble"], 0.65, "core"),
    ("ds_search_bin", "二分查找", "第四章 排序与查找", "在有序序列中折半定位的O(logn)查找算法", [], 0.9, "core"),
]

_NET = [
    ("net_osi", "OSI七层模型", "第一章 网络体系结构", "物理/数据链路/网络/传输/会话/表示/应用七层参考模型", [], 0.85, "foundational"),
    ("net_tcp_ip", "TCP/IP四层模型", "第一章 网络体系结构", "网络接口/网络层/传输层/应用层工程实现模型", ["net_osi"], 0.85, "foundational"),
    ("net_encapsulation", "封装与解封装", "第一章 网络体系结构", "每层加首部封装发送、逐层剥离解封装接收", ["net_osi"], 0.7, "core"),
    ("net_mac_ip", "MAC与IP地址", "第一章 网络体系结构", "链路层物理地址与网络层逻辑地址的区别与协作", ["net_osi"], 0.75, "foundational"),
    ("net_arp", "ARP地址解析", "第二章 网络层", "IP到MAC的广播解析协议，维护ARP缓存表", ["net_mac_ip"], 0.8, "core"),
    ("net_ip_class", "IP分类与子网", "第二章 网络层", "ABCDE五类地址与子网掩码CIDR划分", ["net_mac_ip"], 0.85, "core"),
    ("net_ip_fragment", "IP分片与重组", "第二章 网络层", "MTU限制下的IP数据报分片传输与目的端重组", ["net_ip_class"], 0.7, "core"),
    ("net_icmp", "ICMP控制报文", "第二章 网络层", "差错报告与探测的网络层辅助协议，ping基于它", ["net_ip_class"], 0.65, "core"),
    ("net_routing", "路由算法", "第二章 网络层", "距离矢量与链路状态两大类路由算法", ["net_ip_class"], 0.85, "advanced"),
    ("net_nat", "NAT网络地址转换", "第二章 网络层", "内网私有地址到公网地址的端口映射转换", ["net_ip_class"], 0.7, "core"),
    ("net_ipv6", "IPv6", "第二章 网络层", "128位地址下一代IP协议，头部简化支持自动配置", ["net_ip_class"], 0.6, "core"),
    ("net_udp", "UDP协议", "第三章 传输层", "无连接不可靠数据报协议，开销小速度快", ["net_tcp_ip"], 0.8, "core"),
    ("net_tcp", "TCP协议", "第三章 传输层", "面向连接可靠的字节流协议，流控与拥塞控制", ["net_udp"], 0.95, "core"),
    ("net_three_hand", "三次握手", "第三章 传输层", "TCP建立连接的SYN-SYNACK-ACK三次交互", ["net_tcp"], 0.9, "core"),
    ("net_four_wave", "四次挥手", "第三章 传输层", "TCP断开连接的FIN-ACK四步与TIME_WAIT", ["net_tcp"], 0.85, "core"),
    ("net_flow_ctrl", "TCP流量控制", "第三章 传输层", "滑动窗口机制实现的收端到发端流量控制", ["net_tcp"], 0.85, "advanced"),
    ("net_congestion", "TCP拥塞控制", "第三章 传输层", "慢启动/拥塞避免/快重传/快恢复的AIMD算法", ["net_flow_ctrl"], 0.9, "advanced"),
    ("net_sliding_window", "滑动窗口", "第三章 传输层", "允许连续发送未确认帧的流水线机制", ["net_flow_ctrl"], 0.8, "advanced"),
    ("net_dns", "DNS域名解析", "第四章 应用层", "层次化域名到IP的分布式解析系统", ["net_udp"], 0.85, "core"),
    ("net_http", "HTTP协议", "第四章 应用层", "无状态请求-响应的Web应用层文本协议", ["net_tcp"], 0.9, "core"),
    ("net_https", "HTTPS与TLS", "第四章 应用层", "HTTP over TLS，非对称协商对称密钥的加密传输", ["net_http"], 0.85, "advanced"),
    ("net_http2", "HTTP/2多路复用", "第四章 应用层", "二进制分帧支持单连接多流并行的HTTP新版", ["net_http"], 0.6, "advanced"),
    ("net_cookie_session", "Cookie与Session", "第四章 应用层", "客户端Cookie与服务端Session两种状态保持机制", ["net_http"], 0.75, "core"),
    ("net_cdn", "CDN内容分发", "第四章 应用层", "边缘节点缓存就近分发的架构，降低延迟骨干减压", ["net_dns"], 0.65, "core"),
    ("net_socket", "Socket编程", "第五章 网络编程", "应用层到传输层的BSD Socket API编程接口", ["net_tcp"], 0.8, "core"),
    ("net_io_model", "I/O多路复用", "第五章 网络编程", "select/poll/epoll同时监听多路描述符就绪事件", ["net_socket"], 0.85, "advanced"),
    ("net_reactor", "Reactor模式", "第五章 网络编程", "事件驱动+非阻塞IO的服务器并发设计模式", ["net_io_model"], 0.7, "advanced"),
    ("net_proxy", "代理与反向代理", "第五章 网络编程", "正向代理替客户端请求、反向代理替服务端接入", ["net_http"], 0.65, "core"),
    ("net_load_balance", "负载均衡", "第五章 网络编程", "轮询/最少连接/哈希等流量分发策略", ["net_proxy"], 0.7, "core"),
    ("net_qos", "服务质量QoS", "第五章 网络编程", "带宽/延迟/抖动/丢包率的网络质量保障体系", ["net_tcp"], 0.55, "core"),
]

_DB = [
    ("db_dbms", "DBMS三层模式", "第一章 数据库基础", "外模式/模式/内模式三层与两级映像的数据独立性", [], 0.8, "foundational"),
    ("db_data_model", "数据模型", "第一章 数据库基础", "层次/网状/关系/对象数据模型的演进", ["db_dbms"], 0.7, "foundational"),
    ("db_relational", "关系模型", "第一章 数据库基础", "以二维表表达实体与关系的数据模型，数学基础", ["db_data_model"], 0.85, "core"),
    ("db_key", "键与完整性", "第一章 数据库基础", "主键/外键/候选键与实体完整性/参照完整性", ["db_relational"], 0.8, "core"),
    ("db_relational_alg", "关系代数", "第一章 数据库基础", "选择/投影/连接/除等集合运算的形式化查询语言", ["db_relational"], 0.7, "advanced"),
    ("db_sql_ddl", "SQL-DDL", "第二章 SQL语言", "CREATE/ALTER/DROP定义表结构与约束", ["db_relational"], 0.85, "foundational"),
    ("db_sql_dml", "SQL-DML", "第二章 SQL语言", "INSERT/UPDATE/DELETE/SELECT数据操作语句", ["db_sql_ddl"], 0.95, "foundational"),
    ("db_join", "连接查询", "第二章 SQL语言", "内连接/左外/右外/全外/交叉连接的表间关联", ["db_sql_dml"], 0.9, "core"),
    ("db_subquery", "子查询", "第二章 SQL语言", "嵌套在WHERE/HAVING的SELECT，相关与非相关子查询", ["db_sql_dml"], 0.85, "core"),
    ("db_group", "分组聚合", "第二章 SQL语言", "GROUP BY/HAVING分组与COUNT/SUM/AVG聚合", ["db_sql_dml"], 0.85, "core"),
    ("db_view", "视图", "第二章 SQL语言", "基于基表导出的虚表，简化查询与逻辑独立性", ["db_sql_ddl", "db_sql_dml"], 0.75, "core"),
    ("db_index", "索引", "第三章 存储与索引", "B+树/哈希索引加速查询的空间换时间结构", ["db_sql_ddl"], 0.9, "core"),
    ("db_index_design", "索引设计", "第三章 存储与索引", "选择性/覆盖索引/最左前缀的索引优化原则", ["db_index"], 0.85, "advanced"),
    ("db_transaction", "事务ACID", "第四章 事务与并发", "原子性/一致性/隔离性/持久性的事务四大特性", [], 0.9, "core"),
    ("db_concurrency", "并发控制", "第四章 事务与并发", "锁机制与MVCC实现的读写并发控制", ["db_transaction"], 0.9, "advanced"),
    ("db_lock", "锁与封锁", "第四章 事务与并发", "共享锁/排他锁/意向锁的表级行级封锁协议", ["db_concurrency"], 0.85, "advanced"),
    ("db_isolation", "隔离级别", "第四章 事务与并发", "读未提交/读已提交/可重复读/串行化四级隔离", ["db_concurrency"], 0.9, "advanced"),
    ("db_deadlock_db", "死锁检测", "第四章 事务与并发", "等待图检测环路与超时回滚的死锁处理", ["db_lock"], 0.75, "advanced"),
    ("db_recovery", "日志与恢复", "第五章 恢复", "WAL先写日志/redo/undo的故障恢复机制", ["db_transaction"], 0.85, "advanced"),
    ("db_normalization", "范式理论", "第六章 设计理论", "1NF/2NF/3NF/BCNF函数依赖消除冗余的规范化", ["db_relational"], 0.9, "advanced"),
    ("db_er", "ER模型设计", "第六章 设计理论", "实体-联系图到关系模式的数据库设计方法", ["db_data_model"], 0.8, "core"),
    ("db_denormalization", "反规范化", "第六章 设计理论", "为性能牺牲范式引入冗余的优化策略", ["db_normalization"], 0.65, "advanced"),
    ("db_stored_proc", "存储过程", "第七章 高级特性", "预编译存储于服务端的批处理SQL过程", ["db_sql_dml"], 0.6, "core"),
    ("db_trigger", "触发器", "第七章 高级特性", "事件驱动自动执行的数据库端逻辑", ["db_sql_ddl"], 0.65, "core"),
    ("db_partition", "分区与分表", "第七章 高级特性", "水平/垂直分区与分库分表的分布式扩展", ["db_index"], 0.7, "advanced"),
    ("db_optim", "查询优化", "第七章 高级特性", "基于代价的优化器CBO与执行计划分析", ["db_index_design"], 0.8, "advanced"),
    ("db_nosql", "NoSQL", "第八章 新技术", "键值/文档/列族/图四大类非关系数据库", ["db_relational"], 0.65, "core"),
    ("db_cap", "CAP与BASE", "第八章 新技术", "一致性/可用性/分区容忍三选二与最终一致", ["db_nosql"], 0.75, "advanced"),
    ("db_distributed_tx", "分布式事务", "第八章 新技术", "两阶段提交2PC与三阶段提交3PC的分布式协议", ["db_transaction"], 0.8, "advanced"),
    ("db_mvcc", "MVCC多版本", "第八章 新技术", "多版本并发控制实现读不阻塞写的快照隔离", ["db_concurrency"], 0.8, "advanced"),
]

_CO = [
    ("co_cpu", "CPU指令周期", "第一章 计算机概论", "取指/译码/执行/访存/写回的CPU工作周期", [], 0.85, "foundational"),
    ("co_bus", "总线结构", "第一章 计算机概论", "数据/地址/控制三总线与总线仲裁", [], 0.75, "foundational"),
    ("co_num_rep", "数据表示", "第二章 数据表示", "原码/反码/补码的带符号数表示与运算", [], 0.8, "foundational"),
    ("co_float", "浮点数IEEE754", "第二章 数据表示", "符号/阶码/尾数的浮点数标准与精度范围", ["co_num_rep"], 0.85, "advanced"),
    ("co_alu", "运算器ALU", "第二章 数据表示", "加法器为核心实现算术逻辑运算的部件", ["co_num_rep"], 0.8, "core"),
    ("co_instruction", "指令系统", "第三章 指令与CPU", "操作码+地址码的机器指令格式与寻址方式", ["co_cpu"], 0.85, "core"),
    ("co_addressing", "寻址方式", "第三章 指令与CPU", "立即/直接/间接/寄存器/基址等指令寻址", ["co_instruction"], 0.8, "core"),
    ("co_cisc_risc", "CISC与RISC", "第三章 指令与CPU", "复杂指令集与精简指令集两种CPU设计哲学", ["co_instruction"], 0.7, "core"),
    ("co_pipeline", "流水线技术", "第三章 指令与CPU", "指令重叠执行的吞吐率提升技术", ["co_instruction"], 0.9, "advanced"),
    ("co_hazard", "流水线冲突", "第三章 指令与CPU", "结构/数据/控制冒险的检测与处理", ["co_pipeline"], 0.9, "advanced"),
    ("co_forwarding", "数据旁路", "第三章 指令与CPU", "前递/旁路解决数据冒险的硬件优化", ["co_hazard"], 0.75, "advanced"),
    ("co_branch_pred", "分支预测", "第三章 指令与CPU", "静态/动态预测减少控制冒险停顿", ["co_hazard"], 0.8, "advanced"),
    ("co_register", "寄存器组", "第三章 指令与CPU", "通用/专用/程序计数器/状态寄存器", ["co_instruction"], 0.7, "core"),
    ("co_cache", "高速缓存Cache", "第四章 存储系统", "容量小速度快缓解CPU与主存速度差的层级", [], 0.9, "advanced"),
    ("co_cache_map", "Cache映射", "第四章 存储系统", "直接/全相联/组相联三种主存块到Cache行映射", ["co_cache"], 0.9, "advanced"),
    ("co_cache_replace", "Cache替换", "第四章 存储系统", "LRU/FIFO/随机算法的Cache行淘汰", ["co_cache_map"], 0.8, "advanced"),
    ("co_cache_write", "Cache写策略", "第四章 存储系统", "全写法/写回法与写分配/不分配的组合", ["co_cache"], 0.8, "advanced"),
    ("co_mem_hierarchy", "存储层次", "第四章 存储系统", "寄存器/Cache/主存/辅存的金字塔结构", ["co_cache"], 0.75, "core"),
    ("co_vm", "虚拟存储器", "第四章 存储系统", "页式/段式/段页式虚存与TLB地址转换", ["co_mem_hierarchy"], 0.85, "advanced"),
    ("co_tlb", "TLB快表", "第四章 存储系统", "页表项的硬件高速缓存加速虚实地址转换", ["co_vm"], 0.8, "advanced"),
    ("co_interrupt", "中断机制", "第五章 输入输出", "程序/硬件中断的响应/保护/恢复过程", ["co_cpu"], 0.85, "core"),
    ("co_dma", "DMA直接存储", "第五章 输入输出", "外设与主存直接数据交换绕过CPU", ["co_interrupt"], 0.8, "advanced"),
    ("co_io", "I/O控制方式", "第五章 输入输出", "程序查询/中断/DMA/通道四种I/O方式", ["co_interrupt"], 0.8, "core"),
    ("co_disk", "磁盘存储", "第五章 输入输出", "盘片/柱面/扇区物理结构与CHS/LBA寻址", [], 0.65, "core"),
    ("co_raid", "磁盘阵列RAID", "第五章 输入输出", "条带/镜像/校验的冗余磁盘阵列级别", ["co_disk"], 0.7, "advanced"),
    ("co_io_bus", "I/O总线", "第五章 输入输出", "PCI/USB/SCSI等外设总线标准", ["co_bus"], 0.55, "core"),
    ("co_multicore", "多核处理器", "第五章 输入输出", "片上多核与共享内存的并行架构", ["co_pipeline"], 0.7, "advanced"),
    ("co_out_of_order", "乱序执行", "第五章 输入输出", "Tomasulo动态调度与重排序缓冲ROB", ["co_pipeline"], 0.75, "advanced"),
    ("co_amdahl", "Amdahl定律", "第五章 输入输出", "可加速比上限与串行比例的加速定律", [], 0.7, "core"),
    ("co_perf", "性能指标", "第一章 计算机概论", "主频/CPI/MIPS/浮点运算能力的评价指标", ["co_cpu"], 0.65, "foundational"),
]

_ALGO = [
    ("algo_complexity", "渐近复杂度", "第一章 复杂度分析", "大O/Ω/Θ渐近界的算法效率分析记号", [], 0.9, "foundational"),
    ("algo_master", "主定理", "第一章 复杂度分析", "分治递归T(n)=aT(n/b)+f(n)的渐近界求解", ["algo_complexity"], 0.85, "advanced"),
    ("algo_recurrence", "递推求解", "第一章 复杂度分析", "代入法/递归树/主定理解递推方程", ["algo_complexity"], 0.8, "core"),
    ("algo_amortized", "平摊分析", "第一章 复杂度分析", "聚合/会计/势能法的平摊代价界", ["algo_complexity"], 0.75, "advanced"),
    ("algo_divide", "分治法", "第二章 算法设计", "分解-递归-合并的策略如归并排序", ["algo_master"], 0.85, "core"),
    ("algo_dp", "动态规划", "第三章 动态规划", "最优子结构+重叠子问题+备忘录/表的自底向上", ["algo_recurrence"], 0.95, "advanced"),
    ("algo_dp_lcs", "LCS最长公共子序列", "第三章 动态规划", "经典两序列对齐的DP问题", ["algo_dp"], 0.85, "advanced"),
    ("algo_dp_knapsack", "背包问题DP", "第三章 动态规划", "0-1/完全/多重背包的动态规划解", ["algo_dp"], 0.9, "advanced"),
    ("algo_dp_edit_dist", "编辑距离", "第三章 动态规划", "插入/删除/替换三种操作的最小代价DP", ["algo_dp"], 0.8, "advanced"),
    ("algo_greedy", "贪心算法", "第四章 贪心", "局部最优选择策略与贪心选择性质证明", ["algo_divide"], 0.85, "core"),
    ("algo_greedy_huffman", "Huffman编码", "第四章 贪心", "频率贪心构造最优前缀编码树", ["algo_greedy"], 0.8, "core"),
    ("algo_greedy_interval", "区间调度", "第四章 贪心", "活动安排/区间覆盖的经典贪心问题", ["algo_greedy"], 0.75, "core"),
    ("algo_backtrack", "回溯法", "第五章 搜索与回溯", "深度优先试探+剪枝的解空间搜索", ["algo_divide"], 0.85, "core"),
    ("algo_branch_bound", "分支限界", "第五章 搜索与回溯", "广度优先+界限函数剪枝的最优解搜索", ["algo_backtrack"], 0.8, "advanced"),
    ("algo_nqueens", "N皇后", "第五章 搜索与回溯", "回溯法经典约束满足问题", ["algo_backtrack"], 0.7, "core"),
    ("algo_dfs_algo", "图DFS应用", "第六章 图算法", "连通分量/拓扑序/割点桥的DFS框架", ["algo_backtrack"], 0.8, "core"),
    ("algo_scc", "强连通分量", "第六章 图算法", "Kosaraju/Tarjan基于DFS的有向图SCC算法", ["algo_dfs_algo"], 0.85, "advanced"),
    ("algo_bfs_algo", "图BFS应用", "第六章 图算法", "无权最短路/层序遍历的BFS框架", ["algo_divide"], 0.75, "core"),
    ("algo_dfs_cut", "割点与桥", "第六章 图算法", "Tarjan基于dfn/low的关节点与桥判定", ["algo_dfs_algo"], 0.8, "advanced"),
    ("algo_network_flow", "网络流", "第六章 图算法", "最大流最小割定理与Ford-Fulkerson增广", [], 0.85, "advanced"),
    ("algo_kmp", "KMP匹配", "第七章 字符串", "next数组避免回溯的O(n)串匹配算法", ["algo_complexity"], 0.85, "advanced"),
    ("algo_manacher", "Manacher回文", "第七章 字符串", "利用回文对称性的O(n)最长回文子串", ["algo_kmp"], 0.7, "advanced"),
    ("algo_trie_search", "AC自动机", "第七章 字符串", "Trie+失败指针的多模式串匹配", ["algo_kmp"], 0.75, "advanced"),
    ("algo_suffix", "后缀数组", "第七章 字符串", "sa/rank数组与LCP的字符串处理利器", ["algo_kmp"], 0.7, "advanced"),
    ("algo_segment_tree", "线段树", "第八章 数据结构进阶", "区间查询/修改的O(logn)平衡二叉树", ["algo_divide"], 0.85, "advanced"),
    ("algo_bit", "树状数组BIT", "第八章 数据结构进阶", "lowbit前缀和的O(logn)区间统计结构", ["algo_complexity"], 0.8, "advanced"),
    ("algo_lca", "最近公共祖先LCA", "第八章 数据结构进阶", "倍增/Tarjan/树剖求树节点LCA", ["algo_dfs_algo"], 0.75, "advanced"),
    ("algo_hash_algo", "字符串哈希", "第七章 字符串", "多项式滚动哈希的串匹配与去重", ["algo_complexity"], 0.7, "core"),
    ("algo_randomized", "随机化算法", "第九章 高级算法", "Monte Carlo/Las Vegas与随机化快排", ["algo_divide"], 0.65, "advanced"),
    ("algo_np", "NP完全性", "第九章 高级算法", "P/NP/NPC/NP-hard的归约与判定", ["algo_complexity"], 0.8, "advanced"),
]

_DM = [
    ("dm_prop_logic", "命题逻辑", "第一章 数理逻辑", "命题与联结词非且或蕴涵等价的形式系统", [], 0.85, "foundational"),
    ("dm_pred_logic", "谓词逻辑", "第一章 数理逻辑", "量词全称存在与谓词的一阶逻辑系统", ["dm_prop_logic"], 0.85, "advanced"),
    ("dm_inference", "推理规则", "第一章 数理逻辑", "假言推理/归谬/CP规则的形式推导", ["dm_prop_logic"], 0.8, "core"),
    ("dm_resolution", "归结原理", "第一章 数理逻辑", "合取范式消解对偶的自动定理证明方法", ["dm_inference"], 0.8, "advanced"),
    ("dm_set", "集合论", "第二章 集合论", "并交补差幂集与笛卡尔积的基本运算", [], 0.8, "foundational"),
    ("dm_relation", "二元关系", "第三章 关系与函数", "自反/对称/传递/反自反/反对称的性质", ["dm_set"], 0.9, "core"),
    ("dm_equiv", "等价关系", "第三章 关系与函数", "自反+对称+传递的划分等价类关系", ["dm_relation"], 0.85, "core"),
    ("dm_order", "偏序与全序", "第三章 关系与函数", "自反+反对称+传递的偏序与线性序", ["dm_relation"], 0.8, "core"),
    ("dm_closure", "关系闭包", "第三章 关系与函数", "自反/对称/传递闭包的最小扩充", ["dm_relation"], 0.8, "advanced"),
    ("dm_function", "函数映射", "第三章 关系与函数", "单射/满射/双射的函数分类与复合", ["dm_relation"], 0.85, "core"),
    ("dm_cardinality", "基数", "第三章 关系与函数", "可数/不可数集与势的比较", ["dm_function"], 0.7, "advanced"),
    ("dm_graph_dm", "图论基础", "第四章 图论", "顶点边握手定理/度/路径/回路概念", [], 0.8, "core"),
    ("dm_euler", "欧拉图", "第四章 图论", "一笔画问题与欧拉回路/欧拉路径判定", ["dm_graph_dm"], 0.8, "core"),
    ("dm_hamilton", "哈密顿图", "第四章 图论", "经过每点恰一次的回路与NP完全性", ["dm_graph_dm"], 0.75, "advanced"),
    ("dm_planar", "平面图", "第四章 图论", "无交叉画法与欧拉公式V-E+F=2", ["dm_graph_dm"], 0.8, "advanced"),
    ("dm_color", "图着色", "第四章 图论", "相邻顶点不同色的着色与色数", ["dm_graph_dm"], 0.75, "advanced"),
    ("dm_tree_dm", "树的概念", "第四章 图论", "无回路连通图与生成树计数", ["dm_graph_dm"], 0.8, "core"),
    ("dm_bipartite", "二部图", "第四章 图论", "顶点集二划分无边同部与匹配判定", ["dm_graph_dm"], 0.75, "core"),
    ("dm_group", "群与子群", "第五章 代数系统", "运算+结合律+幺元+逆元的代数结构", ["dm_set"], 0.85, "advanced"),
    ("dm_ring", "环与域", "第五章 代数系统", "加法群+乘法半群的环与无零因子域", ["dm_group"], 0.8, "advanced"),
    ("dm_homomorphism", "同态与同构", "第五章 代数系统", "保运算映射的代数结构对应关系", ["dm_group"], 0.8, "advanced"),
    ("dm_lattice", "格与布尔代数", "第五章 代数系统", "上下确界运算的偏序格与布尔代数", ["dm_order"], 0.75, "advanced"),
    ("dm_count_sum", "加法乘法原理", "第六章 组合计数", "分类相加/分步相乘的基本计数原则", [], 0.8, "foundational"),
    ("dm_perm", "排列组合", "第六章 组合计数", "有序排列P与无序组合C的计数公式", ["dm_count_sum"], 0.9, "core"),
    ("dm_inclusion", "容斥原理", "第六章 组合计数", "并集计数加减交替的多集合计数", ["dm_perm"], 0.8, "advanced"),
    ("dm_pigeonhole", "鸽巢原理", "第六章 组合计数", "n+1物入n穴必有重复的存在性证明", ["dm_count_sum"], 0.75, "core"),
    ("dm_gen_func", "生成函数", "第七章 递推与生成函数", "幂级数编码数列的计数与递推求解", ["dm_perm"], 0.8, "advanced"),
    ("dm_recurrence_dm", "递推关系", "第七章 递推与生成函数", "线性常系数齐次递推的特征根解法", ["dm_gen_func"], 0.8, "advanced"),
    ("dm_catalan", "Catalan数", "第七章 递推与生成函数", "栈序列/二叉树/括号化的组合计数", ["dm_recurrence_dm"], 0.7, "advanced"),
    ("dm_probability", "离散概率", "第七章 递推与生成函数", "古典概型与条件概率贝叶斯的离散计算", ["dm_perm"], 0.75, "core"),
]

_COMP = [
    ("comp_lexer", "词法分析", "第二章 词法分析", "有限自动机DFA/NFA识别token的扫描器", [], 0.85, "core"),
    ("comp_regex", "正则表达式", "第二章 词法分析", "描述token模式的正则语言与Thompson构造", ["comp_lexer"], 0.75, "core"),
    ("comp_nfa_dfa", "NFA转DFA", "第二章 词法分析", "子集构造法将NFA确定化为DFA", ["comp_lexer"], 0.85, "advanced"),
    ("comp_dfa_min", "DFA最小化", "第二章 词法分析", "Hopcroft等价类合并的最小化算法", ["comp_nfa_dfa"], 0.8, "advanced"),
    ("comp_grammar", "文法与乔姆斯基谱系", "第三章 语法分析", "0型/1型/2型/3型文法分类与产生式", [], 0.85, "core"),
    ("comp_ambiguity", "文法二义性", "第三章 语法分析", "一个串有多棵分析树的文法缺陷与消解", ["comp_grammar"], 0.8, "advanced"),
    ("comp_ll1", "LL(1)分析", "第三章 语法分析", "FIRST/FOLLOW集+预测表的递归下降自顶向下", ["comp_grammar"], 0.9, "advanced"),
    ("comp_first_follow", "FIRST与FOLLOW", "第三章 语法分析", "首符集与后继符集的计算用于预测分析", ["comp_ll1"], 0.85, "advanced"),
    ("comp_lr0", "LR(0)分析", "第三章 语法分析", "项目集规范族+状态机的自底向上移进归约", ["comp_grammar"], 0.9, "advanced"),
    ("comp_slr", "SLR分析", "第三章 语法分析", "用FOLLOW集解决LR(0)归约冲突", ["comp_lr0"], 0.85, "advanced"),
    ("comp_lalr", "LALR分析", "第三章 语法分析", "合并同心集缩小LR(1)状态数的分析器", ["comp_slr"], 0.85, "advanced"),
    ("comp_lr1", "LR(1)分析", "第三章 语法分析", "带向前搜索符的最强LR分析器", ["comp_lalr"], 0.85, "advanced"),
    ("comp_ast", "抽象语法树", "第四章 语义分析", "去除分析树冗余的抽象语法表示", ["comp_grammar"], 0.75, "core"),
    ("comp_attr_grammar", "属性文法", "第四章 语义分析", "S属性/L属性的综合继承语义计算", ["comp_ast"], 0.8, "advanced"),
    ("comp_type_check", "类型检查", "第四章 语义分析", "类型系统与类型推断/转换的静态语义", ["comp_attr_grammar"], 0.75, "core"),
    ("comp_symbol_table", "符号表", "第四章 语义分析", "作用域与嵌套结构的标识符管理", ["comp_ast"], 0.7, "core"),
    ("comp_ir", "中间代码", "第五章 中间代码与优化", "三地址码/四元式/逆波兰式中间表示", ["comp_ast"], 0.8, "core"),
    ("comp_dag", "DAG基本块优化", "第五章 中间代码与优化", "有向无环图的公共子表达式删除", ["comp_ir"], 0.75, "advanced"),
    ("comp_dataflow", "数据流分析", "第五章 中间代码与优化", "到达定值/活跃变量/可用表达式分析", ["comp_dag"], 0.85, "advanced"),
    ("comp_loop_optim", "循环优化", "第五章 中间代码与优化", "代码外提/强度削弱/归纳变量删除", ["comp_dataflow"], 0.8, "advanced"),
    ("comp_target_code", "目标代码生成", "第六章 目标代码生成", "寄存器分配/指令选择/窥孔优化", ["comp_ir"], 0.8, "advanced"),
    ("comp_peephole", "窥孔优化", "第六章 目标代码生成", "局部窗口的冗余指令消除优化", ["comp_target_code"], 0.7, "advanced"),
    ("comp_runtime", "运行时存储", "第七章 运行时环境", "活动记录栈式管理/堆分配/静态区", ["comp_symbol_table"], 0.75, "core"),
    ("comp_activation", "活动记录", "第七章 运行时环境", "函数调用栈帧的参数/局部/返回地址布局", ["comp_runtime"], 0.8, "core"),
    ("comp_storage", "静态与动态", "第七章 运行时环境", "静态/栈式/堆式三种存储分配策略", ["comp_runtime"], 0.7, "core"),
    ("comp_error_recover", "错误恢复", "第三章 语法分析", "恐慌模式/短语级恢复的容错策略", ["comp_lr0"], 0.65, "core"),
    ("comp_yacc", "Yacc/Lex工具", "第八章 工具实践", "Lex词法+Yacc语法自动生成编译器", ["comp_lexer", "comp_lr0"], 0.6, "core"),
    ("comp_backpatch", "回填技术", "第五章 中间代码与优化", "布尔表达式与控制流的跳转回填", ["comp_ir"], 0.7, "advanced"),
    ("comp_gc", "垃圾回收", "第七章 运行时环境", "标记清除/复制/分代的内存回收", ["comp_runtime"], 0.7, "advanced"),
    ("comp_sdd", "语法制导定义", "第四章 语义分析", "SDD属性文法与SDT翻译方案", ["comp_attr_grammar"], 0.75, "advanced"),
]

_SE = [
    ("se_lifecycle", "软件生命周期", "第一章 软件工程概述", "需求/设计/编码/测试/维护的瀑布迭代过程", [], 0.85, "foundational"),
    ("se_waterfall", "瀑布模型", "第二章 过程模型", "线性顺序阶段流的经典开发模型", ["se_lifecycle"], 0.8, "foundational"),
    ("se_agile", "敏捷开发", "第二章 过程模型", "迭代增量/拥抱变化的敏捷宣言与Scrum", ["se_waterfall"], 0.85, "core"),
    ("se_spiral", "螺旋模型", "第二章 过程模型", "风险驱动螺旋迭代的过程模型", ["se_waterfall"], 0.75, "core"),
    ("se_incremental", "增量模型", "第二章 过程模型", "分批交付核心+逐步扩展的开发模型", ["se_waterfall"], 0.7, "core"),
    ("se_requirement", "需求工程", "第三章 需求分析", "功能/非功能需求获取与分析规约", [], 0.85, "core"),
    ("se_use_case", "用例图", "第三章 需求分析", "参与者+用例+关系的UML行为建模", ["se_requirement"], 0.8, "core"),
    ("se_user_story", "用户故事", "第三章 需求分析", "作为X我想Y以便Z的敏捷需求表达", ["se_requirement"], 0.7, "core"),
    ("se_srs", "需求规约SRS", "第三章 需求分析", "结构化需求规约文档与IEEE标准", ["se_requirement"], 0.7, "core"),
    ("se_design", "软件设计", "第四章 软件设计", "概要设计与详细设计的设计活动", ["se_requirement"], 0.8, "core"),
    ("se_cohesion", "内聚耦合", "第四章 软件设计", "高内聚低耦合的模块化设计度量原则", ["se_design"], 0.85, "advanced"),
    ("se_modularity", "模块化", "第四章 软件设计", "分解与信息隐蔽的模块划分原则", ["se_design"], 0.8, "core"),
    ("se_class_diagram", "类图", "第四章 软件设计", "UML类/属性/关联/多态的静态结构建模", ["se_design"], 0.8, "core"),
    ("se_sequence", "顺序图", "第四章 软件设计", "时序消息交互的UML行为图", ["se_class_diagram"], 0.75, "core"),
    ("se_state_chart", "状态图", "第四章 软件设计", "状态转换的UML行为建模", ["se_class_diagram"], 0.7, "core"),
    ("se_design_pattern", "设计模式", "第五章 设计模式", "创建/结构/行为三大类23种经典模式", ["se_modularity"], 0.9, "advanced"),
    ("se_singleton", "单例模式", "第五章 设计模式", "全局唯一实例的创建型模式", ["se_design_pattern"], 0.75, "core"),
    ("se_factory", "工厂方法", "第五章 设计模式", "封装对象创建的创建型模式", ["se_design_pattern"], 0.75, "core"),
    ("se_observer", "观察者模式", "第五章 设计模式", "发布订阅的行为型模式", ["se_design_pattern"], 0.75, "core"),
    ("se_strategy", "策略模式", "第五章 设计模式", "算法族可互换的行为型模式", ["se_design_pattern"], 0.75, "core"),
    ("se_testing", "软件测试", "第六章 测试与维护", "单元/集成/系统/验收的V模型测试", ["se_design"], 0.85, "core"),
    ("se_black_box", "黑盒测试", "第六章 测试与维护", "等价类划分/边界值的功能测试", ["se_testing"], 0.8, "core"),
    ("se_white_box", "白盒测试", "第六章 测试与维护", "语句/分支/路径覆盖的结构测试", ["se_testing"], 0.8, "core"),
    ("se_regression", "回归测试", "第六章 测试与维护", "变更后重新验证未受影响功能", ["se_testing"], 0.75, "core"),
    ("se_ci_cd", "CI/CD", "第六章 测试与维护", "持续集成/持续交付的自动化流水线", ["se_testing"], 0.8, "advanced"),
    ("se_version", "版本控制", "第六章 测试与维护", "Git分支模型与代码协作管理", ["se_lifecycle"], 0.7, "core"),
    ("se_code_review", "代码审查", "第六章 测试与维护", "同伴评审发现缺陷提升质量", ["se_version"], 0.65, "core"),
    ("se_refactor", "重构", "第五章 设计模式", "不改外部行为改善内部结构的代码优化", ["se_design_pattern"], 0.75, "advanced"),
    ("se_solid", "SOLID原则", "第五章 设计模式", "单一职责/开闭/里氏替换等面向对象原则", ["se_modularity"], 0.8, "advanced"),
    ("se_devops", "DevOps", "第六章 测试与维护", "开发运维一体化的文化与工具实践", ["se_ci_cd"], 0.7, "advanced"),
]

_SEC = [
    ("sec_cia", "安全三要素CIA", "第一章 安全基础", "机密性/完整性/可用性的安全目标三元组", [], 0.85, "foundational"),
    ("sec_threat_model", "威胁建模", "第一章 安全基础", "STRIDE/DREAD等威胁识别评估方法", ["sec_cia"], 0.8, "core"),
    ("sec_sym_crypto", "对称加密", "第二章 密码学", "AES/DES分组密码与流密码的单密钥加密", [], 0.85, "core"),
    ("sec_asym_crypto", "非对称加密", "第二章 密码学", "RSA/ECC公钥加密体系与密钥交换", ["sec_sym_crypto"], 0.9, "advanced"),
    ("sec_hash", "哈希函数", "第二章 密码学", "MD5/SHA抗碰撞单向散列与完整性校验", [], 0.85, "core"),
    ("sec_mac", "消息认证码MAC", "第二章 密码学", "HMAC密钥哈希的消息真实性验证", ["sec_hash"], 0.8, "core"),
    ("sec_signature", "数字签名", "第二章 密码学", "非对称加密+哈希的不可否认签名机制", ["sec_asym_crypto", "sec_hash"], 0.9, "advanced"),
    ("sec_pkc", "公钥基础设施PKI", "第三章 PKI与证书", "CA证书链与X.509的信任体系", ["sec_signature"], 0.85, "advanced"),
    ("sec_tls", "TLS协议", "第三章 PKI与证书", "握手协商对称密钥的传输层安全协议", ["sec_pkc", "sec_asym_crypto"], 0.9, "advanced"),
    ("sec_key_exchange", "密钥交换", "第二章 密码学", "Diffie-Hellman密钥协商与离散对数", ["sec_asym_crypto"], 0.85, "advanced"),
    ("sec_random", "随机数与PRNG", "第二章 密码学", "真随机与伪随机数生成器的密码学安全", ["sec_hash"], 0.7, "core"),
    ("sec_access", "访问控制", "第四章 访问控制", "DAC/MAC/RBAC的权限管理模型", ["sec_cia"], 0.8, "core"),
    ("sec_auth", "认证授权", "第四章 访问控制", "身份认证与OAuth2/JWT的授权机制", ["sec_access"], 0.85, "core"),
    ("sec_least_priv", "最小权限", "第四章 访问控制", "最小化授权原则的安全设计", ["sec_access"], 0.7, "core"),
    ("sec_firewall", "防火墙", "第五章 网络安全", "包过滤/状态检测/应用层的访问控制网关", ["sec_cia"], 0.8, "core"),
    ("sec_ids", "入侵检测IDS", "第五章 网络安全", "特征/异常检测的入侵发现系统", ["sec_firewall"], 0.75, "advanced"),
    ("sec_vpn", "VPN隧道", "第五章 网络安全", "IPsec/SSL VPN的加密隧道技术", ["sec_tls"], 0.7, "core"),
    ("sec_ddos", "DDoS防护", "第五章 网络安全", "流量清洗与Anycast的分布式拒绝服务防护", ["sec_firewall"], 0.75, "advanced"),
    ("sec_sql_inject", "SQL注入", "第六章 Web安全", "拼接SQL导致非授权执行的注入攻击与参数化防御", [], 0.9, "core"),
    ("sec_xss", "XSS跨站脚本", "第六章 Web安全", "注入恶意脚本的跨站攻击与转义防御", ["sec_sql_inject"], 0.85, "core"),
    ("sec_csrf", "CSRF跨站请求", "第六章 Web安全", "伪造用户请求的跨站攻击与Token防御", ["sec_xss"], 0.8, "core"),
    ("sec_buffer_overflow", "缓冲区溢出", "第七章 系统安全", "越界写覆盖返回地址的栈溢出与NX/ASLR防御", [], 0.85, "advanced"),
    ("sec_rce", "远程代码执行", "第七章 系统安全", "反序列化/命令注入的RCE漏洞", ["sec_buffer_overflow"], 0.8, "advanced"),
    ("sec_malware", "恶意软件", "第七章 系统安全", "病毒/木马/蠕虫/勒索的分类与特征", [], 0.7, "core"),
    ("sec_reverse", "逆向工程", "第七章 系统安全", "反汇编/反编译的静态动态分析技术", ["sec_buffer_overflow"], 0.7, "advanced"),
    ("sec_sandbox", "沙箱隔离", "第七章 系统安全", "进程级隔离的执行环境与逃逸", ["sec_access"], 0.65, "advanced"),
    ("sec_owasp", "OWASP Top10", "第六章 Web安全", "Web应用最严重十大风险清单", ["sec_xss", "sec_csrf"], 0.8, "core"),
    ("sec_zero_trust", "零信任架构", "第四章 访问控制", "永不信任持续验证的零信任安全模型", ["sec_auth"], 0.7, "advanced"),
    ("sec_side_channel", "侧信道攻击", "第八章 高级安全", "时间/功耗/缓_flush侧信道泄漏", ["sec_sym_crypto"], 0.7, "advanced"),
    ("sec_crypto_protocol", "密码协议", "第三章 PKI与证书", "零知识证明/安全多方计算/同态加密", ["sec_signature"], 0.75, "advanced"),
]

_COURSES_DATA = {
    "ds": _DS, "net": _NET, "db": _DB, "co": _CO, "algo": _ALGO,
    "dm": _DM, "comp": _COMP, "se": _SE, "sec": _SEC,
}

# 第 11-30 门扩展课程（curriculum_ext）：同构数据合并，install() 自动装库
from .curriculum_ext import COURSES_EXT, COURSE_ROWS_EXT  # noqa: E402
COURSES.update(COURSES_EXT)
_COURSES_DATA.update(COURSE_ROWS_EXT)


# 每难度的题干模板 + 参考答案模板（结合考点名与summary）
_TIER_TMPL = {
    "foundation": [
        ("简述{}的基本概念。", "{}：{}"),
        ("{}的定义是什么？", "{}是{}。"),
        ("举例说明{}在计算机学科中的含义。", "{}指{}。"),
    ],
    "consolidation": [
        ("对比{}与其相关概念，说明核心区别。", "{}的核心要点在于：{}；区别体现在适用场景与实现方式不同。"),
        ("分析{}的关键性质并说明其意义。", "{}的关键性质：{}，其意义在于保证数据正确性与系统效率。"),
        ("为什么需要{}？说明解决的问题。", "{}解决的问题是{}，通过特定机制提升整体性能。"),
    ],
    "extension": [
        ("给定一个场景，分析如何用{}解决实际工程问题。", "运用{}：先分析问题约束，再依据{}（{}）设计方案，权衡时空复杂度。"),
        ("{}的底层原理是什么？若性能不足如何优化？", "{}底层依赖{}；优化方向为减少额外开销与利用局部性原理。"),
        ("论述{}与其他方法的优劣权衡及适用边界。", "{}优势是高效且语义清晰，劣势是特定条件受限（{}），需按场景权衡选用。"),
    ],
}


def build_course_concepts(course: str, rows: list) -> list[Concept]:
    return [Concept(cid=cid, name=name, course=course, chapter=ch, summary=summ,
                    prerequisites=prereqs, exam_weight=ew, tier=t, noise="core")
            for (cid, name, ch, summ, prereqs, ew, t) in rows]


def build_course_quiz(course: str, concepts: list[Concept]) -> list[QuizItem]:
    items: list[QuizItem] = []
    n = 0
    for c in concepts:
        for tier_key in ("foundation", "consolidation", "extension"):
            q_tmpl, a_tmpl = _TIER_TMPL[tier_key][n % 3]
            items.append(QuizItem(
                qid=f"bank_{c.cid}_{tier_key}_{n}",
                cid=c.cid, course=course, tier=tier_key,
                question=q_tmpl.format(c.name),
                reference_answer=a_tmpl.format(c.name, c.summary),
                source_quote=c.summary[:40], grounded=True))
            n += 1
    return items


def install(keep_existing: bool = True) -> dict:
    """构建全量课程库并合并落盘。

    keep_existing=True 时保留现有 concepts.json/quiz_bank.json 中已存在的条目
    （自动补 course 字段为 'os'），只追加 9 门新课程。False 则全量重建。
    """
    from .. import storage

    if keep_existing:
        existing_concepts = storage.load_concepts()
        existing_quiz = storage.load_quiz_bank()
        # 补 course 字段（旧数据无此字段，默认 os）
        for c in existing_concepts:
            if not getattr(c, "course", None):
                c.course = "os"
        for q in existing_quiz:
            if not getattr(q, "course", None):
                q.course = "os"
        existing_cids = {c.cid for c in existing_concepts}
        existing_qids = {q.qid for q in existing_quiz}
    else:
        existing_concepts, existing_quiz = [], []
        existing_cids, existing_qids = set(), set()

    new_concepts = list(existing_concepts)
    new_quiz = list(existing_quiz)
    stats = {"courses": {}, "total_concepts": len(new_concepts),
              "total_quiz": len(new_quiz)}
    for course, rows in _COURSES_DATA.items():
        cs = build_course_concepts(course, rows)
        qs = build_course_quiz(course, cs)
        added_c = sum(1 for c in cs if c.cid not in existing_cids)
        added_q = sum(1 for q in qs if q.qid not in existing_qids)
        new_concepts.extend(c for c in cs if c.cid not in existing_cids)
        new_quiz.extend(q for q in qs if q.qid not in existing_qids)
        existing_cids.update(c.cid for c in cs)
        existing_qids.update(q.qid for q in qs)
        stats["courses"][course] = {"concepts": added_c, "quiz": added_q}

    storage.save_concepts(new_concepts)
    storage.save_quiz_bank(new_quiz)
    stats["total_concepts"] = len(new_concepts)
    stats["total_quiz"] = len(new_quiz)
    return stats


if __name__ == "__main__":
    s = install()
    print(f"课程库构建完成：{s['total_concepts']} 考点 / {s['total_quiz']} 题")
    for course, st in s["courses"].items():
        print(f"  {COURSES.get(course, course):8s}  +{st['concepts']:2d} 考点  +{st['quiz']:2d} 题")
