import React from 'react';
import { Box, AppBar, Toolbar, Typography, IconButton, useTheme } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import Sidebar from '../Sidebar';

interface LayoutProps {
    children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    const [sidebarOpen, setSidebarOpen] = React.useState(true);
    const theme = useTheme();

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            {/* Sidebar */}
            <Sidebar open={sidebarOpen} onClose={toggleSidebar} />

            {/* Main content */}
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    bgcolor: theme.palette.background.default,
                }}
            >
                {/* Top AppBar */}
                <AppBar
                    position="fixed"
                    sx={{
                        width: `calc(100% - ${sidebarOpen ? 240 : 0}px)`,
                        ml: `${sidebarOpen ? 240 : 0}px`,
                        transition: theme.transitions.create(['width', 'margin'], {
                            easing: theme.transitions.easing.sharp,
                            duration: theme.transitions.duration.leavingScreen,
                        }),
                    }}
                >
                    <Toolbar>
                        <IconButton
                            color="inherit"
                            aria-label="toggle sidebar"
                            onClick={toggleSidebar}
                            edge="start"
                            sx={{ mr: 2 }}
                        >
                            <MenuIcon />
                        </IconButton>

                        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                            Crypto Trading Bot
                        </Typography>

                        <IconButton color="inherit">
                            <NotificationsIcon />
                        </IconButton>
                        <IconButton color="inherit">
                            <AccountCircleIcon />
                        </IconButton>
                    </Toolbar>
                </AppBar>

                {/* Content area */}
                <Box
                    component="div"
                    sx={{
                        flexGrow: 1,
                        p: 3,
                        mt: 8,
                        bgcolor: theme.palette.background.default,
                        transition: theme.transitions.create('margin', {
                            easing: theme.transitions.easing.sharp,
                            duration: theme.transitions.duration.leavingScreen,
                        }),
                    }}
                >
                    {children}
                </Box>
            </Box>
        </Box>
    );
};

export default Layout; 