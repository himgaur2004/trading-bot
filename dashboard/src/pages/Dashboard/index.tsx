import React from 'react';
import {
    Grid,
    Card,
    CardContent,
    Typography,
    Box,
    useTheme,
    CircularProgress,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    AccountBalance as AccountBalanceIcon,
    ShowChart as ShowChartIcon,
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

interface MetricCardProps {
    title: string;
    value: string | number;
    change?: number;
    icon: React.ReactNode;
    loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
    title,
    value,
    change,
    icon,
    loading = false,
}) => {
    const theme = useTheme();

    return (
        <Card>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6" color="textSecondary">
                        {title}
                    </Typography>
                    <Box sx={{ color: theme.palette.primary.main }}>{icon}</Box>
                </Box>
                {loading ? (
                    <CircularProgress size={24} />
                ) : (
                    <>
                        <Typography variant="h4" component="div">
                            {value}
                        </Typography>
                        {change !== undefined && (
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    mt: 1,
                                    color: change >= 0 ? 'success.main' : 'error.main',
                                }}
                            >
                                {change >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                                <Typography variant="body2" sx={{ ml: 1 }}>
                                    {Math.abs(change)}%
                                </Typography>
                            </Box>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );
};

const Dashboard: React.FC = () => {
    const theme = useTheme();

    // Sample data for charts
    const equityCurveData = {
        labels: Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`),
        datasets: [
            {
                label: 'Portfolio Value',
                data: Array.from({ length: 30 }, () => Math.random() * 1000 + 10000),
                borderColor: theme.palette.primary.main,
                backgroundColor: `${theme.palette.primary.main}20`,
                fill: true,
                tension: 0.4,
            },
        ],
    };

    const drawdownData = {
        labels: Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`),
        datasets: [
            {
                label: 'Drawdown',
                data: Array.from({ length: 30 }, () => -(Math.random() * 10)),
                borderColor: theme.palette.error.main,
                backgroundColor: `${theme.palette.error.main}20`,
                fill: true,
                tension: 0.4,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
        },
        scales: {
            x: {
                grid: {
                    display: false,
                },
            },
            y: {
                grid: {
                    color: theme.palette.divider,
                },
            },
        },
    };

    return (
        <Box sx={{ p: 3 }}>
            {/* Metrics Overview */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Total Balance"
                        value="$12,345.67"
                        change={2.5}
                        icon={<AccountBalanceIcon />}
                    />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Daily PnL"
                        value="$234.56"
                        change={1.2}
                        icon={<ShowChartIcon />}
                    />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Win Rate"
                        value="65%"
                        icon={<TrendingUpIcon />}
                    />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Open Positions"
                        value="3"
                        icon={<ShowChartIcon />}
                    />
                </Grid>
            </Grid>

            {/* Charts */}
            <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Equity Curve
                            </Typography>
                            <Box sx={{ height: 300 }}>
                                <Line data={equityCurveData} options={chartOptions} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} md={4}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Drawdown
                            </Typography>
                            <Box sx={{ height: 300 }}>
                                <Line data={drawdownData} options={chartOptions} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Recent Activity */}
            <Grid container spacing={3} sx={{ mt: 3 }}>
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Recent Activity
                            </Typography>
                            {/* Add recent activity table or list here */}
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default Dashboard; 